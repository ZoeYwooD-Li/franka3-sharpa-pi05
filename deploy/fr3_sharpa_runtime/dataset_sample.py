"""Read LeRobot RGB samples and convert named converter joints to policy order."""

import io
import json
from pathlib import Path

import numpy as np

from .protocol import FR3_POLICY_NAMES
from .protocol import LEFT_SHARPA_NAMES
from .protocol import RIGHT_SHARPA_NAMES

ORDER = (*FR3_POLICY_NAMES[:7], *LEFT_SHARPA_NAMES, *FR3_POLICY_NAMES[7:], *RIGHT_SHARPA_NAMES)


def canonical_positions(values, names):
    if len(names) != len(set(names)) or len(values) != len(names):
        raise ValueError("Source joint names must be unique and match vector length")
    expected = [name + ".position" for name in ORDER]
    if not set(expected).issubset(names):
        raise ValueError("Source state is missing required named joint positions")
    state = np.asarray(values, dtype=np.float32)[[names.index(name) for name in expected]]
    if not np.isfinite(state).all():
        raise ValueError("Source joint positions must be finite")
    return state


def load_sample(path: Path, prompt: str, frame: int = 0) -> dict:
    from PIL import Image
    import pyarrow.parquet as pq

    path = path.resolve()
    table = pq.read_table(path)
    if not 0 <= frame < table.num_rows:
        raise ValueError("Sample frame outside episode")
    row = table.slice(frame, 1).to_pylist()[0]
    dataset = next((parent for parent in path.parents if (parent / "meta/info.json").is_file()), None)
    info = json.loads((dataset / "meta/info.json").read_text()) if dataset else None
    state = np.asarray(row["observation.state"], dtype=np.float32)
    if info:
        if info["fps"] != 30:
            raise ValueError("This checkpoint requires a 30 Hz dataset")
        names = info["features"]["observation.state"].get("names")
        if names and any(name.endswith(".position") for name in names):
            state = canonical_positions(state, names)
        elif names and list(names) != list(ORDER):
            raise ValueError("Unrecognized source joint order")
    if state.shape != (58,) or not np.isfinite(state).all():
        raise ValueError("Expected canonical 58D state; raw 116D state needs named dataset metadata")

    def image(camera):
        key = f"observation.images.{camera}"
        value = row.get(key)
        if isinstance(value, dict) and value.get("bytes") is not None:
            return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
        if info is None or not info.get("video_path"):
            raise ValueError(f"Missing embedded RGB or LeRobot video metadata: {camera}")
        import cv2

        episode = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        if abs(float(row["timestamp"]) - frame_index / info["fps"]) > 0.001:
            raise ValueError("Source frame/timestamp mismatch")
        video = dataset / info["video_path"].format(
            episode_index=episode, episode_chunk=episode // info["chunks_size"], video_key=key
        )
        capture = cv2.VideoCapture(str(video))
        try:
            if abs(capture.get(cv2.CAP_PROP_FPS) - info["fps"]) > 0.01:
                raise ValueError(f"Video FPS mismatch: {video}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, pixels = capture.read()
            if not success:
                raise ValueError(f"Cannot decode {video} frame {frame_index}")
            return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    return {
        "observation/state": state,
        "observation/image": image("cam0"),
        "observation/left_wrist_image": image("cam2"),
        "observation/right_wrist_image": image("cam1"),
        "prompt": prompt,
    }
