"""The one-repository release must carry the reviewed model source."""

import io

from PIL import Image

from deploy.fr3_sharpa_runtime.trex.contract import IMAGE_SIZE
from deploy.fr3_sharpa_runtime.trex.serve import VENDORED_TREX, _png_black, verify_model_source


def test_bundled_model_source_and_warmup_image():
    verify_model_source(VENDORED_TREX)
    assert Image.open(io.BytesIO(_png_black())).size == IMAGE_SIZE
