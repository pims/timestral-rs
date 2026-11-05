"""Pinned reference loading, public input conversion, and forecast assertions."""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.model import TimestralModel
import timestral.covariate_encoding
import timestral.forecaster

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def source_hash(module):
    return hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()


def verify_reference():
    fixture = json.loads((FIXTURES / "categorical-encoding.json").read_text())
    distribution = importlib.metadata.distribution("timestral")
    provenance = json.loads(distribution.read_text("direct_url.json"))
    if provenance["vcs_info"]["commit_id"] != fixture["reference_commit"]:
        raise RuntimeError("Installed Timestral does not match the pinned reference")
    for module in (timestral.covariate_encoding, timestral.forecaster):
        if source_hash(module) != fixture["source_sha256"][module.__name__]:
            raise RuntimeError(f"Reference source mismatch: {module.__name__}")
    return fixture["reference_commit"]


def load_reference_model(checkpoint):
    verify_reference()
    torch.set_num_threads(1)
    config = TimestralConfig.load(checkpoint / "config.json")
    model = TimestralModel(config).eval()
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    return model


def category_codes(labels):
    """Assign identities across history and future; null remains a missing code."""
    identities = {}
    codes = []
    for label in labels:
        if label is None:
            codes.append(np.nan)
        else:
            identities.setdefault(label, len(identities))
            codes.append(identities[label])
    return codes


def covariate_array(members, group):
    """Pack one covariate group as [batch, covariates, time], in name order."""
    packed = []
    for member in members:
        rows = member.get("covariates", {}).get(group, {})
        if not rows:
            return None
        packed_rows = []
        for name in sorted(rows):
            row = rows[name]
            if group.endswith("past"):
                values = row
            else:
                values = list(row["history"]) + list(row["future"])
            if group.startswith("categorical"):
                values = category_codes(values)
            packed_rows.append(values)
        packed.append(packed_rows)
    dtype = np.float64 if group.startswith("categorical") else np.float32
    return np.asarray(packed, dtype=dtype)


def assert_forecast_matches(actual, expected, history, mask, context, device, description):
    """Compare in observed target units, retaining the stricter median tolerance."""
    history = np.asarray(history, dtype=np.float64)[-context:]
    observed = np.isfinite(history)
    if mask is not None:
        observed &= np.asarray(mask, dtype=bool)[-context:]
    values = history[observed]
    unit = 1.0
    if len(values) > 1:
        unit = max(1.0, float(values.std(ddof=1)))

    # Nonlinear decoding can amplify Metal accumulation differences in the tails.
    # The median must still meet the CPU criterion, including near zero.
    absolute_tolerance = (1e-3 if device == "metal" else 2e-4) * unit
    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=absolute_tolerance, err_msg=description
    )
    np.testing.assert_allclose(
        actual[..., 10], expected[..., 10], rtol=1e-4, atol=2e-4 * unit,
        err_msg=f"{description}/median",
    )
