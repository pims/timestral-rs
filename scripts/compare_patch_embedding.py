"""Compare Rust patch embeddings with the official Python layer, using identical inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.model import TimestralModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("rust_output", type=Path, help=".npy output from the Rust executable")
args = parser.parse_args()
config = TimestralConfig.load(args.checkpoint / "config.json")
if (config.d_model, len(config.block_schedule)) not in ((192, 8), (384, 12)) or config.patch_size != 32:
    parser.error("Expected a Timestral-5M or Timestral-28M configuration")
model = TimestralModel(config).eval()
model.load_state_dict(load_file(args.checkpoint / "model.safetensors"), strict=True)
values = ((torch.arange(96, dtype=torch.float32) - 48) / 32).reshape(1, 3, 32)
features = torch.cat([values, torch.zeros_like(values)], dim=-1)
with torch.inference_mode():
    expected = model.representation.patch_proj(features).numpy()
actual = np.load(args.rust_output, allow_pickle=False)
np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
print(f"Patch embedding matches Python: shape={actual.shape}, max_abs_error={np.max(np.abs(actual - expected)):.3g}")
