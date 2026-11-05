"""Refresh expected values for the inputs in tests/fixtures using pinned Python.

From the repository root:
    python3 tests/fixtures/generate.py

Use --output-dir to inspect regenerated fixtures before replacing checked-in files.
No checkpoint or Rust executable is needed.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import torch
import timestral.covariate_encoding as encoding
import timestral.forecaster as forecasting
import timestral.scaler as scaler

FIXTURES = Path(__file__).resolve().parent


def source_hash(module):
    return hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()


def verify_reference():
    fixture = json.loads((FIXTURES / "categorical-encoding.json").read_text())
    distribution = importlib.metadata.distribution("timestral")
    provenance = json.loads(distribution.read_text("direct_url.json"))
    if provenance["vcs_info"]["commit_id"] != fixture["reference_commit"]:
        raise RuntimeError("Installed Timestral does not match the pinned reference")
    for module in (encoding, forecasting):
        if source_hash(module) != fixture["source_sha256"][module.__name__]:
            raise RuntimeError(f"Reference source mismatch: {module.__name__}")


def nullable(values):
    return [float(value) if np.isfinite(value) else None for value in values]


@torch.inference_mode()
def encoding_fixture():
    fixture = json.loads((FIXTURES / "categorical-encoding.json").read_text())
    fixture["numpy_version"] = np.__version__
    fixture["source_sha256"] = {
        module.__name__: source_hash(module) for module in (encoding, forecasting)
    }
    for case in fixture["cases"]:
        reference = SimpleNamespace(device=torch.device("cpu"), max_context=case["max_context"])
        history = np.asarray(case["history"], dtype=np.float32)
        inputs, mask = forecasting.TimestralForecaster._as_context(
            reference, history, case.get("observation_mask")
        )
        target = forecasting.TimestralForecaster._encoding_target(reference, inputs, mask)
        # Codes represent label identity; null is distinct from unseen labels.
        labels = case["past"] + case.get("future", [])
        identities = dict.fromkeys(label for label in labels if label is not None)
        codes = {label: float(index) for index, label in enumerate(identities)}
        values = [np.nan if label is None else codes[label] for label in labels]
        encoded = forecasting.TimestralForecaster._encode_categorical(
            reference, torch.tensor(values, dtype=torch.float64)[None, None], target
        )[0, 0].numpy()
        length = len(history)
        case["encoding_target"] = nullable(target[0].numpy())
        case["encoded_past"] = nullable(encoded[:length])
        case["encoded_future"] = nullable(encoded[length:]) if "future" in case else None

    pattern = np.asarray(fixture["reduction_pattern"], dtype=np.float32)
    for case in fixture["mean_reductions"]:
        target = np.resize(pattern, case["length"]).astype(np.float64)
        target = np.where(np.isfinite(target), target, 0.0)
        case["expected_sum_bits"] = f"{target.sum().view(np.uint64):016x}"
    return fixture


def normalization_fixture():
    fixture = json.loads((FIXTURES / "categorical-known-normalization.json").read_text())
    fixture.update({
        "torch_version": torch.__version__,
        "machine": platform.machine(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "scaler_sha256": source_hash(scaler),
    })
    for case in fixture["cases"]:
        length = case["length"]
        values = np.full(length, 104.45187, dtype=np.float32)
        if not case["constant"]:
            # Construct in Python before casting, matching the original fixture inputs.
            values = np.asarray([
                104.45187 + (index % 17) * 0.001 for index in range(length)
            ], dtype=np.float32)
        observed = np.asarray([index % 7 != 0 for index in range(length)])
        inputs = torch.from_numpy(values)[None]
        mask = torch.from_numpy(observed)[None]
        location, scale, _ = scaler.instance_norm_stats(inputs, mask, minimum_scale=1e-6)
        normalized = scaler.apply_affine_asinh_masked(
            inputs, location, scale, mask, use_asinh=True
        )[0]
        case.update({
            "mean": location.item(),
            "scale": scale.item(),
            "sum": (inputs * mask).sum().item(),
            "normalized_prefix": normalized[:min(length, 32)].tolist(),
        })
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    verify_reference()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "categorical-encoding.json": encoding_fixture(),
        "categorical-known-normalization.json": normalization_fixture(),
    }
    for name, fixture in fixtures.items():
        destination = args.output_dir / name
        content = json.dumps(fixture, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write_text(content + "\n")
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
