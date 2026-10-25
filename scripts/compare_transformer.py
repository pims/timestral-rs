"""Compare the full transformer with PyTorch, including row-isolation invariants.

Masks and positions are built by the original Python implementation. This checks
transformer execution, not Rust forecast preparation or end-to-end forecasting.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from safetensors.numpy import load_file as load_arrays, save_file as save_arrays
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.layers import TwoStreamMask, _apply_rotary, _sdpa
from timestral.masks import (
    build_forecast_positions, build_forecast_time_mask,
    build_two_stream_positions, build_two_stream_time_mask,
    build_two_stream_variate_mask, compute_enc_key_validity,
)
from timestral.model import TimestralModel


def assert_vector_close(got, want, rtol, atol, message):
    """Bound RMS error per vector, including channels where large terms cancel."""
    assert got.shape == want.shape, f'{message}: shape mismatch'
    assert np.isfinite(got).all(), f'{message}: non-finite output'
    rms = np.sqrt(np.mean(want.astype(np.float64) ** 2, axis=-1, keepdims=True))
    delta_rms = np.sqrt(np.mean((got.astype(np.float64) - want) ** 2, axis=-1, keepdims=True))
    np.testing.assert_array_less(delta_rms, atol + rtol * np.maximum(1, rms),
                                 err_msg=f'{message}: vector RMS error bound')


@torch.inference_mode()
def reference(model, x, positions, time_mask, variable_mask, enc_len):
    cos, sin = model.transformer.rope(x, positions)
    result = dict(cos=cos, sin=sin)
    for i, block in enumerate(model.transformer.layers):
        start = x
        stages = {}
        for j, (norm_layer, attn) in enumerate(zip(block.attn_norms, block.attns)):
            norm = norm_layer(x)
            inputs = norm if attn.is_time else norm.transpose(0, 1)
            q, k, v = attn._qkv(inputs)
            values = dict(attn_norm=norm, q=q, k=k, v=v)
            if attn.is_time:
                q, k = _apply_rotary(q, k, cos, sin)
                values.update(q_rotated=q, k_rotated=k)
            attention = _sdpa(q, k, v, time_mask if attn.is_time else variable_mask)
            gate = attn._gate(inputs).sigmoid() if attn._gate is not None else torch.ones_like(attention)
            gated = attention * gate
            restored = gated if attn.is_time else gated.transpose(0, 1)
            projected = attn.wO(restored)
            x = x + projected
            values.update(attention=attention, gate=gate, gated=gated,
                          attn_projected=projected, attn_residual=x)
            stages.update({f'attn{j}.{name}': tensor for name, tensor in values.items()})
        norm = block.mlp_norm(x)
        gate, up = block.mlp.w12(norm).chunk(2, dim=-1)
        hidden = gate * torch.nn.functional.silu(up)
        mlp_output = block.mlp.w3(hidden)
        explicit = x + mlp_output
        # Compare decomposition with the real block, including split attention.
        mask = TwoStreamMask(time_mask, enc_len) if enc_len else time_mask
        x = block(start, mask, variable_mask, cos, sin)
        # Split and unsplit PyTorch GEMMs can differ at near-zero channels too.
        # Use the same intermediate-vector criterion as the Rust trace below.
        assert_vector_close(explicit.numpy(), x.numpy(), 1e-4, 2e-4,
                            f'Python split/decomposed block {i}')
        stages.update(mlp_norm=norm, mlp_gate=gate, mlp_up=up,
                      mlp_hidden=hidden, mlp_output=mlp_output, output=x)
        result.update({f'layers.{i}.{name}': tensor for name, tensor in stages.items()})
    output = model.transformer.out_norm(x)
    # The caller also checks the complete module below using the original input.
    result['output'] = output
    result['direct_output'] = output
    return result


@torch.inference_mode()
def context_case(model, groups):
    rows = len(groups)
    values = torch.sin(torch.arange(rows * 96, dtype=torch.float32).reshape(rows, 96) / 19)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[-1, :32] = False
    values = values.masked_fill(~observed, 0)
    x = torch.cat([model.representation(values, observed),
                   model.representation.forecast_tokens(rows, 2)], dim=1)
    valid = compute_enc_key_validity(observed.reshape(rows, 3, 32).any(-1), True)
    # Anchor positions at the first observed patch of each group.
    first = torch.zeros(rows, dtype=torch.long)
    for group in set(groups):
        indices = [i for i, value in enumerate(groups) if value == group]
        first[indices] = valid[indices].any(0).long().argmax()
    positions = build_forecast_positions(3, 2, first)
    time = build_forecast_time_mask(3, 2, valid, torch.device('cpu'))
    variable = build_two_stream_variate_mask(torch.tensor(groups), valid, 2)
    return x, positions, time, variable, 3


@torch.inference_mode()
def cases(model):
    yield 'single_row', context_case(model, [0])
    multi = context_case(model, [0, 0, 0])
    yield 'multivariate', multi
    changed = multi[0].clone()
    changed[:, -1] += 0.25
    yield 'future_token_changed', (changed, *multi[1:])
    changed = multi[0].clone()
    changed[1] += 0.25
    yield 'peer_changed', (changed, *multi[1:])
    separate = context_case(model, [0, 0, 1, 1])
    yield 'separate_series', separate
    changed = separate[0].clone()
    changed[2:] += 0.25
    yield 'other_series_changed', (changed, *separate[1:])
    shifted = separate[1] + torch.tensor([[-100], [-100], [32760], [32760]])
    yield 'shifted_positions', (separate[0], shifted, *separate[2:])
    masked_time = separate[2].clone()
    masked_var = separate[3].clone()
    masked_time[0, :, -1] = False
    masked_var[:, :, 1] = False
    yield 'fully_masked_queries', (separate[0], separate[1], masked_time, masked_var, 3)

    # Covariate path: target / past-only / known-future rows for two groups.
    roles = torch.tensor([0, 1, 2, 0, 1, 2])
    known = roles == 2
    values = torch.sin(torch.arange(6 * 160, dtype=torch.float32).reshape(6, 160) / 23)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[~known, 96:] = False
    values = values.masked_fill(~observed, 0)
    x = torch.cat([model.representation(values, observed), model.representation.forecast_tokens(6, 5)], dim=1)
    valid = compute_enc_key_validity(observed.reshape(6, 5, 32).any(-1), True)
    positions = build_two_stream_positions(5, torch.zeros(6, dtype=torch.long))
    time = build_two_stream_time_mask(torch.full((6,), 3), 5, model.config.block_size, valid, known)
    variable = build_two_stream_variate_mask(torch.tensor([0, 0, 0, 1, 1, 1]), valid, 5,
                                            target_rows=roles == 0, bidirectional_rows=known)
    yield 'covariate_masks', (x, positions, time, variable, 5)
    changed = x.clone()
    changed[roles == 0] += 0.25
    yield 'targets_changed', (changed, positions, time, variable, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/examples/transformer'))
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=2e-4)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    binary = args.binary.resolve()
    config = TimestralConfig.load(checkpoint / 'config.json')
    if (config.d_model, len(config.block_schedule)) not in ((192, 8), (384, 12)):
        parser.error('Expected Timestral-5M or Timestral-28M')
    model = TimestralModel(config).eval()
    model.load_state_dict(load_file(checkpoint / 'model.safetensors'), strict=True)
    baselines = {}
    layer_errors = np.zeros(len(config.block_schedule))
    with torch.inference_mode(), tempfile.TemporaryDirectory(prefix='timestral-transformer-') as tmp:
        inputs = Path(tmp) / 'inputs.safetensors'
        output_path = Path(tmp) / 'trace.safetensors'
        for name, (x, positions, time, variable, enc_len) in cases(model):
            expected = reference(model, x, positions, time, variable, enc_len)
            official = model.transformer(x, time, variable, positions)
            torch.testing.assert_close(expected['output'], official, rtol=1e-4, atol=2e-4)
            values = dict(x=x.numpy(), positions=positions.numpy().astype(np.float32),
                          time_allowed=time.numpy().astype(np.uint8))
            if variable is not None:
                values['variable_allowed'] = variable.numpy().astype(np.uint8)
            # The NumPy safetensors writer reads raw memory; transposed variable
            # masks must be made C-contiguous or their rows are silently reordered.
            values = {key: np.ascontiguousarray(value) for key, value in values.items()}
            save_arrays(values, inputs)
            for key, value in load_arrays(inputs).items():
                np.testing.assert_array_equal(value, values[key])
            run = subprocess.run([str(binary), str(checkpoint), str(inputs), str(output_path)],
                                 text=True, capture_output=True)
            if run.returncode:
                raise RuntimeError(f'{name}: {run.stderr}')
            actual = load_arrays(output_path)
            assert set(actual) == set(expected), 'Missing or extra trace stages'
            for stage, tensor in expected.items():
                want = tensor.numpy()
                got = actual[stage]
                if stage in ('cos', 'sin', 'output', 'direct_output'):
                    rtol, atol = (1e-6, 1e-6) if stage in ('cos', 'sin') else (args.rtol, args.atol)
                    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol,
                                               err_msg=f'{name}/{stage}')
                else:
                    # Transformer layers accumulate F32 roundoff. Use a normwise check
                    # on intermediate vectors, where individual channels can
                    # cancel near zero; retain elementwise checks at the public
                    # normalized output and the separate first-block comparison.
                    assert_vector_close(got, want, args.rtol, args.atol, f'{name}/{stage}')
            np.testing.assert_array_equal(actual['direct_output'], actual['output'])
            for i in range(len(config.block_schedule)):
                key = f'layers.{i}.output'
                layer_errors[i] = max(layer_errors[i], float(np.max(np.abs(actual[key] - expected[key].numpy()))))
            if name in ('multivariate', 'separate_series', 'covariate_masks'):
                baselines[name] = actual['output'].copy()
            if name == 'peer_changed':
                assert not np.allclose(actual['output'][0], baselines['multivariate'][0]), 'Variable attention had no effect'
            if name == 'future_token_changed':
                np.testing.assert_allclose(actual['output'][:, :-1], baselines['multivariate'][:, :-1],
                                           rtol=1e-5, atol=1e-5)
            if name == 'other_series_changed':
                np.testing.assert_allclose(actual['output'][:2], baselines['separate_series'][:2], rtol=1e-5, atol=1e-5)
            if name == 'targets_changed':
                rows = [1, 2, 4, 5]
                np.testing.assert_allclose(actual['output'][rows], baselines['covariate_masks'][rows], rtol=1e-5, atol=1e-5)
            if name == 'fully_masked_queries':
                for i, kind in enumerate(config.block_schedule):
                    for j, axis in enumerate(kind.split('_')):
                        if axis == 'variate':
                            assert np.count_nonzero(actual[f'layers.{i}.attn{j}.attention'][:, 1]) == 0
            error = np.max(np.abs(actual['output'] - expected['output'].numpy()))
            print(f'PASS {name}: {len(expected)} stages; normalized output max error {error:.3g}', flush=True)
    print('\nMaximum pre-normalization output error by layer:')
    for i, error in enumerate(layer_errors):
        print(f'  layer {i}: {error:.3g}')


if __name__ == '__main__':
    main()
