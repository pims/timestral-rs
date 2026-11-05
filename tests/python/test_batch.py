"""Independent batches: Python parity, individual Rust calls, isolation, and ordering.

Run through the suite: python tests/python
"""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from timestral.forecaster import TimestralForecaster

from reference import assert_forecast_matches, covariate_array, load_reference_model


def request(covariates=False, missing=False):
    """Three distinct targets; shared labels deliberately have different target means."""
    length, horizon = 97, 65
    time = np.arange(length + horizon, dtype=np.float32)
    members = []
    for index in range(3):
        history = (index * 60 - 30 + 8 * np.sin(time[:length] / (4 + index))).tolist()
        mask = [True] * length
        if missing:
            for position in range(index * 12, index * 12 + 20):
                mask[position] = False
            history[50 + index] = None
        member = {"history": history, "observation_mask": mask}
        if covariates:
            labels = [f"day-{step % 7}" for step in range(length + horizon)]
            # Missing and unseen labels exercise per-member target encoding.
            labels[index * 10] = None
            labels[length + index] = "future-only"
            member["covariates"] = {
                "past": {"rain": np.sin(time[:length] / (5 + index)).tolist()},
                "known": {
                    "trend": {
                        "history": (time[:length] / 100).tolist(),
                        "future": (time[length:] / 100).tolist(),
                    },
                },
                "categorical_past": {"state": labels[:length]},
                "categorical_known": {
                    "day": {"history": labels[:length], "future": labels[length:]},
                },
            }
        members.append(member)
    return {"series": members, "horizon": horizon, "options": {}}


class BatchParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_reference_model(cls.checkpoint)

    def rust(self, data, mode="batch"):
        with tempfile.TemporaryDirectory(prefix="timestral-batch-") as directory:
            directory = Path(directory)
            source, output = directory / "request.json", directory / "forecast.npy"
            source.write_text(json.dumps(data, allow_nan=False))
            command = [str(self.binary.resolve()), str(self.checkpoint.resolve()),
                       str(source), str(output), self.device, mode]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"forecast_device={self.device}", result.stderr)
            return np.load(output)

    @torch.inference_mode()
    def reference(self, data):
        options = data["options"]
        constructor = {key: value for key, value in options.items()
                       if key not in ("symmetric_averaging", "clip_nonnegative")}
        reference = TimestralForecaster(self.model, **constructor)
        members = data["series"]
        history = np.asarray([member["history"] for member in members], dtype=np.float32)
        mask = np.asarray([member["observation_mask"] for member in members], dtype=bool)
        return reference.forecast(
            history, data["horizon"], observation_mask=mask,
            past_covariates=covariate_array(members, "past"),
            known_covariates=covariate_array(members, "known"),
            categorical_past_covariates=covariate_array(members, "categorical_past"),
            categorical_known_covariates=covariate_array(members, "categorical_known"),
            symmetric_averaging=options.get("symmetric_averaging", True),
            clip_nonnegative=options.get("clip_nonnegative", True),
        ).quantiles.numpy()

    def check_forecast(self, data):
        actual = self.rust(data)
        individual = self.rust(data, "individual")
        expected = self.reference(data)
        self.assertEqual(actual.shape, (len(data["series"]), 1, data["horizon"], 21))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((np.diff(actual, axis=-1) >= 0).all())
        for index, member in enumerate(data["series"]):
            context = data["options"].get("max_context", self.model.config.context_length)
            for baseline, name in ((expected, "Python"), (individual, "individual Rust")):
                assert_forecast_matches(
                    actual[index], baseline[index], member["history"],
                    member["observation_mask"], context, self.device,
                    f"member {index}: {name}",
                )
        return actual

    def test_batch_size_one(self):
        data = request(covariates=True)
        data["series"] = data["series"][:1]
        self.check_forecast(data)

    def test_plain_batch_with_missing_values(self):
        data = request(missing=True)
        for index, member in enumerate(data["series"]):
            member["history"][:32 * (index + 1)] = [None] * (32 * (index + 1))
        self.check_forecast(data)

    def test_plain_horizon_beyond_covariate_limit(self):
        data = request()
        data["horizon"] = 321
        self.check_forecast(data)

    def test_covariates_with_missing_values(self):
        self.check_forecast(request(covariates=True, missing=True))

    def test_cropping_and_multiple_blocks(self):
        for covariates in (False, True):
            with self.subTest(covariates=covariates):
                data = request(covariates=covariates, missing=True)
                data["options"] = {
                    "max_context": 65,
                    "per_shot_horizon_patches": 1,
                    "symmetric_averaging": False,
                    "clip_nonnegative": False,
                }
                self.check_forecast(data)

    def test_all_missing_target(self):
        data = request(covariates=True)
        data["series"][1]["history"] = [None] * 97
        self.check_forecast(data)

    def test_member_isolation_and_order(self):
        data = request(covariates=True, missing=True)
        baseline = self.rust(data)
        changed = copy.deepcopy(data)
        changed["series"][1]["history"] = [1000.0] * 97
        changed["series"][1]["covariates"]["known"]["trend"]["future"] = [-500.0] * 65
        changed["series"][1]["covariates"]["categorical_past"]["state"] = ["other"] * 97
        actual = self.rust(changed)
        np.testing.assert_array_equal(actual[[0, 2]], baseline[[0, 2]])
        self.assertFalse(np.array_equal(actual[1], baseline[1]))
        data["series"].reverse()
        np.testing.assert_array_equal(self.rust(data), baseline[::-1])
