"""Compare the head with official PyTorch: projection, softplus, widths and ordering."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from safetensors.numpy import load_file as load_arrays, save_file as save_arrays
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.model import TimestralModel, monotone_quantiles


@torch.inference_mode()
def reference(model, *, tokens=None, raw=None):
    c = model.config
    if tokens is not None:
        raw = model.head.projection(tokens).reshape(*tokens.shape[:2], c.patch_size, c.num_quantiles)
    widths = [0.0] * c.num_quantiles
    m = c.median_index
    for i in range(m):
        widths[i] = c.quantile_knots[i + 1] - c.quantile_knots[i]
    for i in range(m + 1, c.num_quantiles):
        widths[i] = c.quantile_knots[i] - c.quantile_knots[i - 1]
    positive = torch.nn.functional.softplus(raw)
    increments = c.quantile_width_factor * torch.tensor(widths, dtype=torch.float32) * positive
    ordered = monotone_quantiles(raw, c.quantile_knots, c.quantile_width_factor)
    result = dict(raw=raw, softplus=positive, increments=increments, ordered=ordered)
    if tokens is not None:
        quantiles = model.head(tokens)
        result.update(quantiles=quantiles, direct_quantiles=quantiles)
    return result


def assert_ordered(actual, median_index):
    ordered = actual['ordered']
    assert np.isfinite(ordered).all(), 'Non-finite quantiles'
    assert (ordered[..., 1:] >= ordered[..., :-1]).all(), 'Crossing quantiles'
    np.testing.assert_array_equal(ordered[..., median_index], actual['raw'][..., median_index])


def run_binary(binary, checkpoint, values, inputs, output):
    values = {key: np.ascontiguousarray(value) for key, value in values.items()}
    save_arrays(values, inputs)
    for key, value in load_arrays(inputs).items():
        np.testing.assert_array_equal(value, values[key])
    run = subprocess.run([str(binary), str(checkpoint), str(inputs), str(output)], text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    return load_arrays(output)


@torch.inference_mode()
def cases(config):
    generator = torch.Generator().manual_seed(42)
    yield 'zero_tokens', dict(tokens=torch.zeros(1, 2, config.d_model))
    yield 'random_tokens', dict(tokens=torch.randn(2, 3, config.d_model, generator=generator))
    yield 'single_patch', dict(tokens=torch.randn(1, 1, config.d_model, generator=generator))
    yield 'max_decode_block', dict(tokens=torch.randn(1, config.block_size, config.d_model, generator=generator))
    yield 'zero_raw', dict(raw=torch.zeros(2, 32, config.num_quantiles))
    raw = torch.linspace(-1000, 1000, 4 * config.num_quantiles).reshape(1, 4, config.num_quantiles)
    raw[..., config.median_index] = torch.tensor([-100., -1., 0., 100.])
    yield 'extreme_raw', dict(raw=raw)
    # Preserve tiny widths in the negative softplus tail, while keeping the median free.
    raw = torch.tensor([-80., -40., -20., -2., 0., 2., 20.]).reshape(7, 1, 1).repeat(1, 2, config.num_quantiles)
    raw[..., config.median_index] = 0
    yield 'softplus_tails', dict(raw=raw)
    raw = torch.full((2, 2, config.num_quantiles), torch.finfo(torch.float32).max)
    raw[1, :, config.median_index] = -torch.finfo(torch.float32).max
    yield 'finite_saturation', dict(raw=raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/examples/quantile_head'))
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    binary = args.binary.resolve()
    model = TimestralModel(TimestralConfig.load(checkpoint / 'config.json')).eval()
    model.load_state_dict(load_file(checkpoint / 'model.safetensors'), strict=True)
    with tempfile.TemporaryDirectory(prefix='timestral-head-') as tmp:
        inputs, output = Path(tmp) / 'inputs.safetensors', Path(tmp) / 'trace.safetensors'
        for name, values in cases(model.config):
            expected = reference(model, **values)
            actual = run_binary(binary, checkpoint, {k: v.numpy() for k, v in values.items()}, inputs, output)
            assert set(actual) == set(expected), 'Missing or extra trace stage'
            for stage, want in expected.items():
                np.testing.assert_allclose(actual[stage], want.numpy(), rtol=1e-4, atol=2e-4,
                                           err_msg=f'{name}/{stage}')
            assert_ordered(actual, model.config.median_index)
            if name == 'softplus_tails':
                # Ordinary absolute tolerances would conceal tiny values rounded to zero.
                np.testing.assert_allclose(actual['softplus'], expected['softplus'].numpy(), rtol=1e-5, atol=0)
            if 'quantiles' in actual:
                np.testing.assert_array_equal(actual['quantiles'], actual['direct_quantiles'])
            error = np.max(np.abs(actual['ordered'].astype(np.float64) - expected['ordered'].numpy()))
            scale = max(1, float(expected['ordered'].abs().max()))
            print(f'PASS {name}: ordered quantiles, unchanged median; max error {error:.3g} '
                  f'(relative to tensor scale: {error / scale:.3g})', flush=True)


if __name__ == '__main__':
    main()
