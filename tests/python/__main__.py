"""Run fixture checks and Rust/Python comparisons for both checkpoints."""

import argparse
from pathlib import Path
import unittest

import test_batch
import test_covariates
from test_reference_fixtures import ReferenceFixtures


class Covariates5M(test_covariates.CovariateParity):
    pass


class Covariates28M(test_covariates.CovariateParity):
    pass


class Batch5M(test_batch.BatchParity):
    pass


class Batch28M(test_batch.BatchParity):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--binary-dir", type=Path, default=Path("target/release/examples"))
    parser.add_argument("--device", choices=("cpu", "metal"), default="cpu")
    args, remaining = parser.parse_known_args()

    for case, checkpoint, binary in (
        (Covariates5M, "Timestral-5M", "covariate_forecast"),
        (Covariates28M, "Timestral-28M", "covariate_forecast"),
        (Batch5M, "Timestral-5M", "batch_forecast"),
        (Batch28M, "Timestral-28M", "batch_forecast"),
    ):
        case.checkpoint = args.models_dir / checkpoint
        case.binary = args.binary_dir / binary
        case.device = args.device

    unittest.main(argv=[__file__, *remaining], verbosity=2)


if __name__ == "__main__":
    main()
