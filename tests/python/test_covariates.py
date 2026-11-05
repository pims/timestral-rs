"""Covariate forecasts: Python parity and first-block diagnostics.

Run through the suite: python tests/python
"""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file as load_arrays
import torch
from timestral.forecaster import TimestralForecaster
from timestral.scaler import denormalize_quantiles

from reference import (
    assert_forecast_matches,
    covariate_array,
    load_reference_model,
)
from reference_stages import first_block


def numeric_request(length=97, horizon=65, past=True, known=True):
    time = np.arange(length + horizon, dtype=np.float32)
    history = 100 + 0.1 * time[:length] + 8 * np.sin(time[:length] / 4)
    covariates = {"past": {}, "known": {}}
    if past:
        covariates["past"]["rainfall"] = (0.5 + np.sin(time[:length] / 5)) * 4
    if known:
        covariates["known"]["holiday"] = {
            "history": (time[:length] % 7 == 0).astype(np.float32),
            "future": (time[length:] % 7 == 0).astype(np.float32),
        }
    return {"history": history, "horizon": horizon, "covariates": covariates}


def categorical_request(length=97, horizon=65, past=True, known=True, mixed=False):
    time = np.arange(length + horizon, dtype=np.float32)
    history = 100 + 0.1 * time[:length] + 8 * np.sin(time[:length] / 4)
    covariates = {"past": {}, "known": {}, "categorical_past": {}, "categorical_known": {}}
    if past:
        covariates["categorical_past"]["status"] = [
            "normal" if step % 5 else "disrupted" for step in range(length)
        ]
    if known:
        labels = [f"day-{step % 7}" for step in range(length + horizon)]
        covariates["categorical_known"]["weekday"] = {
            "history": labels[:length], "future": labels[length:],
        }
    if mixed:
        # Exercise role packing even when categorical names sort before numeric names.
        covariates["past"]["z_rainfall"] = (0.5 + np.sin(time[:length] / 5)) * 4
        covariates["known"]["z_holiday"] = {
            "history": (time[:length] % 7 == 0).astype(np.float32),
            "future": (time[length:] % 7 == 0).astype(np.float32),
        }
    return {"history": history, "horizon": horizon, "covariates": covariates}


def json_values(value):
    if isinstance(value, np.ndarray):
        return [float(item) if np.isfinite(item) else None for item in value]
    if isinstance(value, dict):
        return {name: json_values(item) for name, item in value.items()}
    return value


def combine_rows(numeric, categorical):
    if categorical is None:
        return numeric
    if numeric is None:
        return categorical.numpy()
    return np.concatenate([numeric, categorical.numpy()], axis=1)


def check_category_encodings(reference, request, codes, trace):
    inputs, mask = reference._as_context(request["history"], None)
    target = reference._encoding_target(inputs, mask)
    encoded = {}
    history_length = len(request["history"])
    for group, values in codes.items():
        tensor = None if values is None else torch.as_tensor(values)
        numeric = reference._encode_categorical(tensor, target)
        encoded[group] = numeric
        if numeric is None:
            continue
        names = sorted(request["covariates"][group])
        for index, name in enumerate(names):
            np.testing.assert_allclose(
                trace[f"{group}:{name}:history"],
                numeric[0, index, :history_length].numpy(),
                rtol=1e-6, atol=1e-6, equal_nan=True,
                err_msg=f"{group}/{name}/history",
            )
            if group == "categorical_known":
                np.testing.assert_allclose(
                    trace[f"{group}:{name}:future"],
                    numeric[0, index, history_length:].numpy(),
                    rtol=1e-6, atol=1e-6, equal_nan=True,
                    err_msg=f"{group}/{name}/future",
                )
    return encoded


def check_block_stages(reference, request, arrays, trace):
    codes = {group: arrays[group] for group in ("categorical_past", "categorical_known")}
    encoded = check_category_encodings(reference, request, codes, trace)
    past = combine_rows(arrays["past"], encoded["categorical_past"])
    known = combine_rows(arrays["known"], encoded["categorical_known"])
    block = first_block(reference, request["history"], request["horizon"], past, known)
    exact_stages = {"key_valid", "time_mask", "variable_mask", "positions"}
    for name, expected in block.tensors.items():
        if name in ("anchor_quantiles", "refined_quantiles"):
            continue
        if name in exact_stages:
            np.testing.assert_array_equal(trace[name], expected.numpy(), err_msg=name)
        else:
            np.testing.assert_allclose(
                trace[name], expected.numpy(), rtol=1e-4, atol=2e-4, err_msg=name
            )

    # Feed identical Rust network outputs into the Python CPU stages. This
    # distinguishes nonlinear arithmetic errors from accumulated network error.
    anchor = denormalize_quantiles(
        torch.from_numpy(trace["normalized_quantiles"]),
        block.location, block.scale, reference.model.config.use_asinh,
    )
    refined = reference._refine_block(
        torch.from_numpy(trace["anchor_quantiles"]),
        block.context, block.observation_mask, None,
    )
    np.testing.assert_allclose(
        trace["anchor_quantiles"], anchor.numpy(), rtol=1e-4, atol=2e-4,
        err_msg="anchor-local",
    )
    np.testing.assert_allclose(
        trace["refined_quantiles"], refined.numpy(), rtol=1e-4, atol=2e-4,
        err_msg="refine-local",
    )


class CovariateParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_reference_model(cls.checkpoint)

    def rust_forecast(self, request, options, trace=False):
        with tempfile.TemporaryDirectory(prefix="timestral-covariates-") as directory:
            directory = Path(directory)
            source = directory / "request.json"
            output = directory / "forecast.npy"
            settings = directory / "options.json"
            stages = directory / "trace.safetensors"
            source.write_text(json.dumps(json_values(request), allow_nan=False))
            settings.write_text(json.dumps(options))
            command = [
                str(self.binary.resolve()), str(self.checkpoint.resolve()),
                str(source), str(output), str(settings),
            ]
            if trace:
                command.append(str(stages))
            command.extend(["--device", self.device])
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"forecast_device={self.device}", result.stderr)
            return np.load(output), load_arrays(stages) if trace else None

    @torch.inference_mode()
    def compare_case(self, request, options=None):
        name = self._testMethodName.removeprefix("test_")
        options = options or {}
        constructor_options = {
            key: value for key, value in options.items()
            if key in ("max_context", "per_shot_horizon_patches",
                       "quantile_real_cap_k", "extreme_value_threshold")
        }
        reference = TimestralForecaster(self.model, **constructor_options)
        arrays = {
            group: covariate_array([request], group)
            for group in ("past", "known", "categorical_past", "categorical_known")
        }
        expected = reference.forecast(
            request["history"], request["horizon"],
            past_covariates=arrays["past"], known_covariates=arrays["known"],
            categorical_past_covariates=arrays["categorical_past"],
            categorical_known_covariates=arrays["categorical_known"],
            symmetric_averaging=options.get("symmetric_averaging", True),
            clip_nonnegative=options.get("clip_nonnegative", True),
        ).quantiles.numpy()
        has_covariates = any(request["covariates"].values())
        actual, trace = self.rust_forecast(request, options, trace=has_covariates)
        if trace is not None:
            check_block_stages(reference, request, arrays, trace)

        self.assertEqual(actual.shape, (1, 1, request["horizon"], 21))
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((np.diff(actual, axis=-1) >= 0).all())
        assert_forecast_matches(
            actual, expected, request["history"], None,
            reference.max_context, self.device, name,
        )
        return actual

    def test_past_only(self):
        request = numeric_request(known=False)
        self.compare_case(request)

    def test_known_binary(self):
        request = numeric_request(past=False)
        self.compare_case(request)

    def test_known_numeric(self):
        request = numeric_request(length=65, past=False)
        request["covariates"]["known"]["holiday"] = {
            "history": np.linspace(0.1, 5, 65, dtype=np.float32),
            "future": np.linspace(5, 10, 65, dtype=np.float32),
        }
        self.compare_case(request)

    def test_combined(self):
        request = numeric_request()
        self.compare_case(request)

    def test_no_symmetry(self):
        request = numeric_request()
        self.compare_case(request, options=dict(symmetric_averaging=False))

    def test_missing_values(self):
        request = numeric_request()
        request['history'][:20] = np.nan
        request['history'][40:70] = np.nan
        request['covariates']['past']['rainfall'][:40] = np.nan
        request['covariates']['known']['holiday']['history'][30:60] = np.nan
        request['covariates']['known']['holiday']['future'][10:25] = np.nan
        self.compare_case(request)

    def test_all_missing_covariates(self):
        request = numeric_request(length=33, horizon=17)
        request['covariates']['past']['rainfall'][:] = np.nan
        for values in request['covariates']['known']['holiday'].values():
            values[:] = np.nan
        self.compare_case(request)

    def test_constant(self):
        request = numeric_request(length=64, horizon=64)
        request['history'][:] = 7
        request['covariates']['past']['rainfall'][:] = 0
        self.compare_case(request)

    def test_future_only_observations(self):
        request = numeric_request(length=33, horizon=65, past=False)
        request['covariates']['known']['holiday']['history'][:] = np.nan
        self.compare_case(request)

    def test_mirrored_future_only(self):
        request = numeric_request(length=33, horizon=65, past=False)
        request['history'] *= -1
        request['covariates']['known']['holiday']['history'][:] = np.nan
        request['covariates']['known']['holiday']['future'] *= -1
        self.compare_case(request, options=dict(symmetric_averaging=False, clip_nonnegative=False))

    def test_signed_target(self):
        request = numeric_request(length=96, horizon=37)
        request['history'] -= 105
        self.compare_case(request)

    def test_native_limit(self):
        request = numeric_request(length=129, horizon=256)
        self.compare_case(request)

    def test_multiblock(self):
        request = numeric_request(length=65, horizon=193)
        self.compare_case(request, options=dict(per_shot_horizon_patches=2))

    def test_cropped_multiblock(self):
        request = numeric_request(length=193, horizon=97)
        self.compare_case(request, options=dict(max_context=65, per_shot_horizon_patches=1))

    def test_tight_cap(self):
        request = numeric_request()
        self.compare_case(request, options=dict(quantile_real_cap_k=0.1))

    def test_extreme_feedback(self):
        request = numeric_request(horizon=97)
        self.compare_case(request, options={
            "per_shot_horizon_patches": 1, "extreme_value_threshold": 0.1,
        })

    def test_no_clipping_or_cap(self):
        request = numeric_request()
        self.compare_case(request, options=dict(clip_nonnegative=False, quantile_real_cap_k=0))

    def test_empty_covariates(self):
        request = numeric_request(past=False, known=False)
        self.compare_case(request)

    def test_categorical_past(self):
        request = categorical_request(known=False)
        self.compare_case(request)

    def test_categorical_known(self):
        request = categorical_request(past=False)
        self.compare_case(request)

    def test_categorical_combined(self):
        request = categorical_request()
        self.compare_case(request)

    def test_categorical_mixed(self):
        request = categorical_request(mixed=True)
        self.compare_case(request)

    def test_categorical_missing_unseen(self):
        request = categorical_request()
        request['history'][:20] = np.nan
        request['history'][40:70] = np.nan
        request['covariates']['categorical_past']['status'][:30] = [None] * 30
        row = request['covariates']['categorical_known']['weekday']
        row['history'][10:40] = [None] * 30
        row['future'][:10] = ['unseen'] * 10
        row['future'][20:30] = [None] * 10
        self.compare_case(request)

    def test_categorical_constant(self):
        request = categorical_request()
        request['history'][:] = 7
        request['covariates']['categorical_past']['status'][:] = ['constant'] * 97
        self.compare_case(request)

    def test_categorical_all_missing(self):
        request = categorical_request()
        request['covariates']['categorical_past']['status'][:] = [None] * 97
        row = request['covariates']['categorical_known']['weekday']
        row['history'][:] = [None] * 97
        row['future'][:] = [None] * 65
        self.compare_case(request)

    def test_categorical_signed(self):
        request = categorical_request()
        request['history'] -= 105
        self.compare_case(request, options=dict(clip_nonnegative=False))

    def test_categorical_reflected(self):
        request = categorical_request()
        request['history'] *= -1
        self.compare_case(request, options=dict(symmetric_averaging=False, clip_nonnegative=False))

    def test_categorical_no_symmetry(self):
        request = categorical_request()
        self.compare_case(request, options=dict(symmetric_averaging=False))

    def test_categorical_multiblock(self):
        request = categorical_request(length=65, horizon=193)
        self.compare_case(request, options=dict(per_shot_horizon_patches=2))

    def test_categorical_cropped_multiblock(self):
        request = categorical_request(length=193, horizon=97, mixed=True)
        row = request['covariates']['categorical_known']['weekday']
        row['history'][:100] = ['outside_window'] * 100
        row['history'][100:125] = [None] * 25
        row['future'][:10] = ['outside_window'] * 10
        self.compare_case(request, options=dict(max_context=65, per_shot_horizon_patches=1))

    def test_categorical_native_limit(self):
        request = categorical_request(length=129, horizon=256)
        self.compare_case(request)

    def test_renaming_categories_preserves_forecast(self):
        request = categorical_request(mixed=True)
        original = self.rust_forecast(request, {})[0]
        covariates = request["covariates"]
        for name, labels in covariates["categorical_past"].items():
            covariates["categorical_past"][name] = ["renamed:" + label for label in labels]
        for row in covariates["categorical_known"].values():
            for part in ("history", "future"):
                row[part] = ["renamed:" + label for label in row[part]]
        renamed = self.compare_case(request)
        np.testing.assert_array_equal(original, renamed)
