"""Integration checks against official Python model tensors with seeded synthetic weights.

Run with the parent project's Python environment and pass the built Rust binary.
This does not download or validate the published checkpoint.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file
from timestral.config import TimestralConfig
from timestral.model import TimestralModel


def check(binary, name, parameter_count):
    torch.manual_seed(42)
    config = TimestralConfig.load(Path(__file__).parent / 'fixtures' / f'config-{name.lower()}.json')
    model = TimestralModel(config).eval()
    state = model.state_dict()
    assert sum(t.numel() for t in state.values()) == parameter_count
    with tempfile.TemporaryDirectory(prefix="timestral-loading-") as tmp:
        directory = Path(tmp)
        config.save(directory / "config.json")
        weights = directory / "model.safetensors"
        save_file(state, weights)
        output = directory / "patch.npy"
        good = subprocess.run([binary, directory, output], text=True, capture_output=True)
        assert good.returncode == 0, good.stderr
        features = ((torch.arange(96, dtype=torch.float32) - 48) / 32).reshape(1, 3, 32)
        features = torch.cat([features, torch.zeros_like(features)], dim=-1)
        with torch.inference_mode():
            expected = model.representation.patch_proj(features).numpy()
        actual = np.load(output)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        print(f"PASS {name}: all weights loaded; Python embedding parity (max error {np.max(np.abs(actual - expected)):.3g})")

        def rejects(tensors, message):
            save_file(tensors, weights)
            result = subprocess.run([binary, directory], text=True, capture_output=True)
            assert result.returncode != 0 and message in result.stderr, result.stderr
            print(f"PASS {name}: rejects {message}")

        tensor_name = f"transformer.layers.{len(config.block_schedule) - 1}.attns.1.wQKV.weight"
        rejects({k: v for k, v in state.items() if k != tensor_name}, "Missing checkpoint tensor")
        rejects({**state, tensor_name: torch.zeros(1)}, "Shape mismatch")
        rejects({**state, tensor_name: state[tensor_name].half()}, "Expected F32")
        rejects({**state, "unexpected.weight": torch.zeros(1)}, "Unexpected checkpoint tensors")
        other = config.to_dict()
        other["d_model"] = 768
        (directory / "config.json").write_text(json.dumps(other))
        result = subprocess.run([binary, directory], text=True, capture_output=True)
        assert result.returncode != 0 and "Unsupported Timestral architecture" in result.stderr, result.stderr
        print(f"PASS {name}: rejects unsupported architectures")
        bad = config.to_dict()
        bad['quantile_width_factor'] = 0
        (directory / "config.json").write_text(json.dumps(bad))
        result = subprocess.run([binary, directory], text=True, capture_output=True)
        assert result.returncode != 0 and "Invalid quantile width factor" in result.stderr, result.stderr
        print(f"PASS {name}: rejects invalid quantile width factor")


if __name__ == '__main__':
    binary = Path(sys.argv[1]).resolve()
    check(binary, '5M', 5_208_960)
    check(binary, '28M', 27_821_184)
