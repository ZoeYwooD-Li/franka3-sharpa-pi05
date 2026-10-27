"""Nominal link0-to-link8 FR3 kinematics and bounded offline IK.

The URDF is the exact file used by the T-Rex data adapter. This module does
not account for calibration error, obstacles, or hand/tool geometry.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class IKError(ValueError):
    """A target cannot be represented safely by the nominal arm model."""


def _xyz(value: str) -> np.ndarray:
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid URDF vector: {value!r}")
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _origin(element) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if element is not None:
        result[:3, 3] = _xyz(element.get("xyz", "0 0 0"))
        roll, pitch, yaw = _xyz(element.get("rpy", "0 0 0"))
        result[:3, :3] = (
            _axis_rotation(np.array([0., 0., 1.]), yaw)
            @ _axis_rotation(np.array([0., 1., 0.]), pitch)
            @ _axis_rotation(np.array([1., 0., 0.]), roll)
        )
    return result


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    delta = target @ actual.T
    vector = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                       delta[1, 0] - delta[0, 1]]) * 0.5
    sine = np.linalg.norm(vector)
    cosine = np.clip((np.trace(delta) - 1) * 0.5, -1., 1.)
    if sine < 1e-10:
        if cosine < 0:
            raise IKError("Near-180-degree orientation target is ambiguous")
        return np.zeros(3)
    angle = np.arctan2(sine, cosine)
    return vector * (angle / sine)


@dataclass(frozen=True)
class _Joint:
    kind: str
    index: int | None
    origin: np.ndarray
    axis: np.ndarray


class ArmKinematics:
    def __init__(self, urdf: str | Path, side: str):
        if side not in ("left", "right"):
            raise ValueError("FR3 side must be left or right")
        base, tip = (f"{side}_fr3v2_link0", f"{side}_fr3v2_link8")
        expected = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]
        root = ET.parse(urdf).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        chain, link, visited = [], tip, set()
        while link != base:
            if link in visited or link not in by_child:
                raise ValueError(f"No URDF chain from {base} to {tip}")
            visited.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.joints = []
        self.lower = np.full(7, -np.inf)
        self.upper = np.full(7, np.inf)
        self.velocity = np.full(7, np.inf)
        used = []
        for joint in reversed(chain):
            kind, name = joint.get("type"), joint.get("name")
            if kind not in ("fixed", "revolute") or joint.find("mimic") is not None:
                raise ValueError(f"Unsupported URDF joint: {name}")
            index = None if kind == "fixed" else expected.index(name)
            axis_element = joint.find("axis")
            axis = _xyz(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0")
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError(f"Zero URDF joint axis: {name}")
            axis = axis / np.linalg.norm(axis)
            if index is not None:
                used.append(index)
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"Missing URDF joint limits: {name}")
                self.lower[index] = float(limit.get("lower"))
                self.upper[index] = float(limit.get("upper"))
                self.velocity[index] = float(limit.get("velocity"))
            self.joints.append(_Joint(kind, index, _origin(joint.find("origin")), axis))
        if sorted(used) != list(range(7)) or not (self.lower < self.upper).all():
            raise ValueError("Unexpected FR3 joints or limits in URDF")

    def _validated_q(self, q) -> np.ndarray:
        value = np.asarray(q, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("FR3 joints must be a finite 7-vector")
        if (value < self.lower - 1e-6).any() or (value > self.upper + 1e-6).any():
            raise IKError("FR3 joints exceed nominal URDF limits")
        return value

    def forward_jacobian(self, q) -> tuple[np.ndarray, np.ndarray]:
        q = self._validated_q(q)
        pose = np.eye(4, dtype=np.float64)
        origins = np.zeros((7, 3), dtype=np.float64)
        axes = np.zeros((7, 3), dtype=np.float64)
        for joint in self.joints:
            pose = pose @ joint.origin
            if joint.index is not None:
                i = joint.index
                origins[i] = pose[:3, 3]
                axes[i] = pose[:3, :3] @ joint.axis
                move = np.eye(4, dtype=np.float64)
                move[:3, :3] = _axis_rotation(joint.axis, q[i])
                pose = pose @ move
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for i in range(7):
            jacobian[:3, i] = np.cross(axes[i], pose[:3, 3] - origins[i])
            jacobian[3:, i] = axes[i]
        return pose, jacobian

    def forward(self, q) -> np.ndarray:
        return self.forward_jacobian(q)[0]

    def inverse(self, target, seed, *, position_tolerance=0.001,
                orientation_tolerance=0.005, max_iterations=100) -> np.ndarray:
        """Solve from a measured/previous seed; reject if residual remains."""
        desired = np.asarray(target, dtype=np.float64)
        if desired.shape != (4, 4) or not np.isfinite(desired).all():
            raise ValueError("IK target must be a finite 4x4 pose")
        if (np.linalg.norm(desired[:3, :3].T @ desired[:3, :3] - np.eye(3)) > 1e-5
                or abs(np.linalg.det(desired[:3, :3]) - 1) > 1e-5):
            raise ValueError("IK target rotation must be orthonormal")
        q = self._validated_q(seed).copy()
        seed = q.copy()
        weight = np.diag([10., 10., 10., 1., 1., 1.])
        for _ in range(max_iterations):
            actual, jacobian = self.forward_jacobian(q)
            position = desired[:3, 3] - actual[:3, 3]
            orientation = _orientation_error(desired[:3, :3], actual[:3, :3])
            if (np.linalg.norm(position) <= position_tolerance
                    and np.linalg.norm(orientation) <= orientation_tolerance):
                return q
            error = np.r_[position, orientation]
            matrix = jacobian.T @ weight @ weight @ jacobian + 0.001 * np.eye(7)
            rhs = jacobian.T @ weight @ weight @ error + 0.0001 * (seed - q)
            step = np.linalg.solve(matrix, rhs)
            magnitude = np.max(np.abs(step))
            if magnitude > 0.2:
                step *= 0.2 / magnitude
            old_cost = np.linalg.norm(weight @ error)
            improved = False
            for factor in (1., 0.5, 0.25, 0.125):
                candidate = np.clip(q + factor * step, self.lower, self.upper)
                reached = self.forward(candidate)
                residual = np.r_[desired[:3, 3] - reached[:3, 3],
                                 _orientation_error(desired[:3, :3], reached[:3, :3])]
                if np.linalg.norm(weight @ residual) < old_cost - 1e-10:
                    q = candidate
                    improved = True
                    break
            if not improved:
                break
        reached = self.forward(q)
        position_error = np.linalg.norm(desired[:3, 3] - reached[:3, 3])
        orientation_error = np.linalg.norm(_orientation_error(desired[:3, :3], reached[:3, :3]))
        raise IKError(f"link8 IK failed: position={position_error:.4f} m, orientation={orientation_error:.4f} rad")
