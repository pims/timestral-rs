"""Validate original-unit univariate block stages and complete Rust forecasts.

Requires the existing Python environment and local official 5M or 28M checkpoint.
The Rust executables perform all forecasting preparation/decoding themselves.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from safetensors.numpy import load_file as load_arrays, save_file as save_arrays
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.forecaster import TimestralForecaster, ffill_with_mask, pad_to_patch
from timestral.masks import build_forecast_positions, build_forecast_time_mask, compute_enc_key_validity
from timestral.model import TimestralModel
from timestral.scaler import denormalize_quantiles


def write_arrays(path, values):
    values = {k: np.ascontiguousarray(v) for k, v in values.items()}
    save_arrays(values, path)
    for key, actual in load_arrays(path).items():
        np.testing.assert_array_equal(actual, values[key])


def run(command):
    result = subprocess.run([str(v) for v in command], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result


@torch.inference_mode()
def check_blocks(model, checkpoint, binary, directory):
    base = (10 * torch.sin(torch.arange(96) / 9) + torch.arange(96) * 0.1).float()
    missing = base.clone()
    missing[:33] = torch.nan
    missing[64:80] = torch.nan
    examples = [
        ('ordinary', base, 2),
        ('one_patch', base, 1),
        ('missing_patches', missing, 3),
        ('constant', torch.full((64,), 7.0), 2),
        ('all_missing', torch.full((64,), torch.nan), 2),
        ('large_offset', 100000 + base / 10, 4),
        ('full_decode_block', base, 32),
    ]
    forecaster = TimestralForecaster(model)
    for name, values, horizon_patches in examples:
        observed = torch.isfinite(values)[None]
        context = torch.where(observed, values[None], 0.0)
        context = ffill_with_mask(context, observed)
        context, observed = pad_to_patch(context, observed, model.config.patch_size)
        patches = context.shape[1] // model.config.patch_size
        scaled, loc, scale = model.scale(context, observed)
        validity = compute_enc_key_validity(observed.reshape(1, patches, 32).any(-1), model.config.attend_missing_patches)
        first = observed.reshape(1, patches, 32).any(-1).long().argmax(-1)
        positions = build_forecast_positions(patches, horizon_patches, first)
        time_mask = build_forecast_time_mask(patches, horizon_patches, validity, torch.device('cpu'))
        features = torch.cat([scaled.reshape(1, patches, 32), (~observed).reshape(1, patches, 32).float()], -1)
        embedded = model.representation(scaled, observed)
        normalized = model._run_two_stream(embedded, model.representation.forecast_tokens(1, horizon_patches), time_mask, None, positions)
        anchor = model.predict_block(context, observed, torch.zeros(1, dtype=torch.long), horizon_patches)
        refined = forecaster._refine_block(anchor, context, observed, None)
        expected = dict(scaled=scaled, loc=loc, scale=scale, key_valid=validity.to(torch.uint8),
                        positions=positions.float(), time_mask=time_mask.to(torch.uint8), patch_features=features,
                        embedded=embedded, normalized_quantiles=normalized, anchor_quantiles=anchor, refined_quantiles=refined)
        inputs, output = directory / 'block-input.safetensors', directory / 'block-output.safetensors'
        write_arrays(inputs, dict(context=context[0].numpy(), observed=observed[0].numpy().astype(np.uint8)))
        run([binary, checkpoint, inputs, horizon_patches, output])
        actual = load_arrays(output)
        assert set(actual) == set(expected)
        stage_local = {}
        if name == 'full_decode_block':
            # This deliberately unbounded diagnostic can produce 1e20-scale tails.
            # sinh and recursive frame updates amplify small network roundoff.
            # Keep the strict end-to-end normalized-network comparison, then
            # verify each nonlinear CPU stage against Python with identical inputs.
            # Complete public forecasts below still use end-to-end comparisons.
            stage_local['anchor_quantiles'] = denormalize_quantiles(
                torch.from_numpy(actual['normalized_quantiles']),
                loc[:, -1:, None], scale[:, -1:, None], model.config.use_asinh)
            stage_local['refined_quantiles'] = forecaster._refine_block(
                torch.from_numpy(actual['anchor_quantiles']), context, observed, None)
        for stage, tensor in expected.items():
            tensor = stage_local.get(stage, tensor)
            if stage in ('key_valid', 'positions', 'time_mask'):
                np.testing.assert_array_equal(actual[stage], tensor.numpy(), err_msg=f'block/{name}/{stage}')
            else:
                np.testing.assert_allclose(actual[stage], tensor.numpy(), rtol=1e-4, atol=2e-4,
                                           err_msg=f'block/{name}/{stage}')
        error = np.max(np.abs(actual['refined_quantiles'] - refined.numpy()))
        relative_scale = max(1, float(refined.abs().max()))
        print(f'PASS block/{name}: {len(expected)} stages; max error {error:.3g} '
              f'(relative to tensor scale: {error / relative_scale:.3g})'
              + ('; nonlinear stages checked with identical inputs' if stage_local else ''), flush=True)


def cases():
    t = np.arange(257, dtype=np.float32)
    demand = 100 + t * 0.05 + 8 * np.sin(t / 4)
    signed = 5 * np.sin(t / 9) + 0.01 * t
    yield 'default', demand[:129], 65, {}, None
    yield 'signed_no_symmetry', signed[:96], 96, dict(symmetric_averaging=False), None
    missing = signed[:97].copy()
    missing[:20] = np.nan
    missing[33:48] = np.nan
    missing[80] = np.inf
    yield 'missing_values', missing, 37, {}, None
    mask = np.ones(96, dtype=np.uint8)
    mask[10:40] = 0
    yield 'explicit_mask', demand[:96], 33, {}, mask
    yield 'constant', np.full(64, 7, dtype=np.float32), 64, {}, None
    yield 'zero', np.zeros(33, dtype=np.float32), 1, {}, None
    yield 'one_observation', np.array([3], dtype=np.float32), 17, {}, None
    yield 'all_missing', np.full(64, np.nan, dtype=np.float32), 33, {}, None
    yield 'forced_blocks', signed[:192], 129, dict(per_shot_horizon_patches=2), None
    yield 'long_horizon', demand[:96], 1057, {}, None
    yield 'context_truncation', signed, 97, dict(max_context=65, per_shot_horizon_patches=1), None
    yield 'tight_cap', demand[:96], 65, dict(quantile_real_cap_k=0.1), None
    yield 'cap_disabled', signed[:96], 65, dict(quantile_real_cap_k=0, clip_nonnegative=False), None
    yield 'extreme_feedback', demand[:96], 65, dict(per_shot_horizon_patches=1, extreme_value_threshold=0.1), None
    yield 'nonnegative_clipping', demand[:96], 65, dict(symmetric_averaging=False), None
    yield 'clip_disabled', demand[:96], 65, dict(clip_nonnegative=False), None
    prefix_negative = np.concatenate([np.array([-7], dtype=np.float32), demand[:128]])
    yield 'clipping_uses_full_history', prefix_negative, 33, dict(max_context=64), None


@torch.inference_mode()
def check_forecasts(model, checkpoint, binary, directory):
    worst = 0.0
    for name, history, horizon, options, mask in cases():
        constructor_keys = ('max_context', 'per_shot_horizon_patches', 'quantile_real_cap_k', 'extreme_value_threshold')
        forecaster = TimestralForecaster(model, **{k: v for k, v in options.items() if k in constructor_keys})
        expected = forecaster.forecast(history, prediction_length=horizon, observation_mask=mask,
            symmetric_averaging=options.get('symmetric_averaging', True),
            clip_nonnegative=options.get('clip_nonnegative', True)).quantiles.numpy()
        inputs, output, settings = directory / 'input.safetensors', directory / 'output.safetensors', directory / 'options.json'
        values = dict(history=history.astype(np.float32))
        if mask is not None:
            values['observed'] = mask
        write_arrays(inputs, values)
        settings.write_text(json.dumps(options))
        run([binary, checkpoint, inputs, horizon, output, settings])
        actual = load_arrays(output)['quantiles']
        assert actual.shape == (1, 1, horizon, 21)
        assert np.isfinite(actual).all(), f'{name}: non-finite output'
        assert (actual[..., 1:] >= actual[..., :-1]).all(), f'{name}: crossing quantiles'
        # In original units, the absolute floor must follow the input's scale.
        # Network/stage checks above retain the normalized-space 2e-4 floor.
        finite = history[np.isfinite(history)]
        context_scale = max(1.0, float(np.std(finite.astype(np.float64)))) if finite.size else 1.0
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4 * context_scale,
                                   err_msg=f'forecast/{name}')
        error = float(np.max(np.abs(actual - expected)))
        median_error = float(np.max(np.abs(actual[..., 10] - expected[..., 10])))
        tensor_scale = max(1, float(np.max(np.abs(expected))))
        worst = max(worst, error)
        print(f'PASS forecast/{name}: {horizon} steps; max error {error:.3g}, '
              f'median max error {median_error:.3g}, relative to tensor scale {error / tensor_scale:.3g}', flush=True)
    # Exercise CSV I/O (one value per line, all 21 output columns plus the step).
    csv_input, csv_output = directory / 'history.csv', directory / 'forecast.csv'
    csv_input.write_text('1\n2\nnan\n3\n')
    run([binary, checkpoint, csv_input, 3, csv_output])
    table = np.loadtxt(csv_output, delimiter=',', skiprows=1)
    assert table.shape == (3, 22) and np.isfinite(table).all()
    np.testing.assert_array_equal(table[:, 0], np.arange(3))
    print('PASS CSV input/output', flush=True)
    write_arrays(directory / 'invalid.safetensors', dict(history=np.array([], dtype=np.float32)))
    result = subprocess.run([str(binary), str(checkpoint), str(directory / 'invalid.safetensors'), '1', str(csv_output)], text=True, capture_output=True)
    assert result.returncode != 0 and 'History and horizon' in result.stderr
    print('PASS rejects empty history', flush=True)
    print(f'\nMaximum complete-forecast error: {worst:.3g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/examples/forecast'))
    parser.add_argument('--block-binary', type=Path, default=Path('target/debug/examples/forecast_block'))
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    model = TimestralModel(TimestralConfig.load(checkpoint / 'config.json')).eval()
    model.load_state_dict(load_file(checkpoint / 'model.safetensors'), strict=True)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='timestral-forecast-') as tmp:
        check_blocks(model, checkpoint, args.block_binary.resolve(), Path(tmp))
        check_forecasts(model, checkpoint, args.binary.resolve(), Path(tmp))


if __name__ == '__main__':
    main()
