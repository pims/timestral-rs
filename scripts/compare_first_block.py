"""Run a Rust first-block executable and compare every stage with official PyTorch.

Uses local 5M or 28M weights. Inputs and traces live in a temporary directory. This is
layer validation, not a full forecast or a performance benchmark.
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
from timestral.masks import build_forecast_positions, build_forecast_time_mask
from timestral.model import TimestralModel


@torch.inference_mode()
def reference(model, x, positions, allowed, enc_len):
    block = model.transformer.layers[0]
    assert len(block.attns) == 1 and block.attns[0].is_time
    attn = block.attns[0]
    norm = block.attn_norms[0](x)
    q, k, v = attn._qkv(norm)
    cos, sin = model.transformer.rope(x, positions)
    qr, kr = _apply_rotary(q, k, cos, sin)
    attention = _sdpa(qr, kr, v, allowed)
    gate = attn._gate(norm).sigmoid() if attn._gate is not None else torch.ones_like(attention)
    gated = attention * gate
    projected = attn.wO(gated)
    residual = x + projected
    mlp_norm = block.mlp_norm(residual)
    mlp_gate, mlp_up = block.mlp.w12(mlp_norm).chunk(2, dim=-1)
    hidden = mlp_gate * torch.nn.functional.silu(mlp_up)
    mlp_output = block.mlp.w3(hidden)
    output = residual + mlp_output
    # Check the decomposed reference against the actual block, including its
    # split context/forecast execution path when there is an encoder split.
    mask = TwoStreamMask(allowed, enc_len) if enc_len else allowed
    official = block(x, mask, None, cos, sin)
    torch.testing.assert_close(output, official, rtol=1e-4, atol=1e-5)
    return dict(attn_norm=norm, q=q, k=k, v=v, cos=cos, sin=sin,
                q_rotated=qr, k_rotated=kr, attention=attention, gate=gate,
                gated=gated, attn_projected=projected, attn_residual=residual,
                mlp_norm=mlp_norm, mlp_gate=mlp_gate, mlp_up=mlp_up,
                mlp_hidden=hidden, mlp_output=mlp_output, output=official)


@torch.inference_mode()
def cases(model):
    # Three context patches and two forecast tokens, with distinct input rows.
    values = ((torch.arange(192, dtype=torch.float32) - 48) / 32).reshape(2, 96)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[1, :32] = False
    values = values.masked_fill(~observed, 0)
    embedded = model.representation(values, observed)
    x = torch.cat([embedded, model.representation.forecast_tokens(2, 2)], dim=1)
    valid = observed.reshape(2, 3, 32).any(-1)
    allowed = build_forecast_time_mask(3, 2, valid, torch.device('cpu'))
    positions = build_forecast_positions(3, 2, torch.tensor([0, 1]))
    yield 'context_and_forecast', x, positions, allowed, 3
    changed = x.clone()
    changed[:, -1] += 0.25
    yield 'future_token_changed', changed, positions, allowed, 3
    # Explicitly exercise negative positions and large position offsets.
    yield 'shifted_positions', x, positions + torch.tensor([[-100], [32760]]), allowed, 3
    masked = allowed.clone()
    masked[0, :, -1, :] = False
    masked[1] = False
    yield 'fully_masked_queries', x, positions, masked, 3
    yield 'zero_input', torch.zeros_like(x), positions, allowed, 3
    yield 'single_token', x[:1, :1], positions[:1, :1], torch.ones(1, 1, 1, 1, dtype=torch.bool), 0
    generator = torch.Generator().manual_seed(42)
    random = torch.randn(2, 11, model.config.d_model, generator=generator)
    causal = torch.ones(11, 11, dtype=torch.bool).tril()[None, None].repeat(2, 1, 1, 1)
    yield 'longer_causal', random, torch.arange(11).repeat(2, 1), causal, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/examples/first_block'))
    parser.add_argument('--rtol', type=float, default=1e-4)
    # Matmul/reduction ordering differs between runtimes. The random-input stress
    # case amplifies tiny upstream differences in the trained MLP, particularly
    # where large contributions cancel near zero. Keep the relative check tight
    # and allow a small absolute floor; print all measured errors below.
    parser.add_argument('--atol', type=float, default=2e-4)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    binary = args.binary.resolve()
    config = TimestralConfig.load(checkpoint / 'config.json')
    if (config.d_model, len(config.block_schedule)) not in ((192, 8), (384, 12)) or config.block_schedule[0] != 'time':
        parser.error('Expected Timestral-5M or Timestral-28M with a time-only first layer')
    model = TimestralModel(config).eval()
    model.load_state_dict(load_file(checkpoint / 'model.safetensors'), strict=True)
    stage_max = {}
    baseline_output = None
    with tempfile.TemporaryDirectory(prefix='timestral-first-block-') as tmp:
        inputs = Path(tmp) / 'inputs.safetensors'
        output = Path(tmp) / 'trace.safetensors'
        for name, x, positions, allowed, enc_len in cases(model):
            expected = reference(model, x, positions, allowed, enc_len)
            save_arrays(dict(x=x.numpy(), positions=positions.numpy().astype(np.float32),
                             allowed=allowed.numpy().astype(np.uint8)), inputs)
            run = subprocess.run([str(binary), str(checkpoint), str(inputs), str(output)],
                                 text=True, capture_output=True)
            if run.returncode:
                raise RuntimeError(f'{name}: {run.stderr}')
            actual = load_arrays(output)
            assert set(actual) == set(expected), 'Missing or extra trace stages'
            for stage, tensor in expected.items():
                want = tensor.numpy()
                got = actual[stage]
                # Small CPU-generated RoPE tables should agree much more closely
                # than accumulated block outputs. This catches the Metal trig
                # regression at large positions even if block tolerances change.
                rtol, atol = (1e-6, 1e-6) if stage in ('cos', 'sin') else (args.rtol, args.atol)
                np.testing.assert_allclose(got, want, rtol=rtol, atol=atol,
                                           err_msg=f'{name}/{stage}')
                stage_max[stage] = max(stage_max.get(stage, 0), float(np.max(np.abs(got - want))))
            if name == 'fully_masked_queries':
                assert np.count_nonzero(actual['attention'][0, -1]) == 0
                assert np.count_nonzero(actual['attention'][1]) == 0
            if name == 'context_and_forecast':
                baseline_output = actual['output'].copy()
            if name == 'future_token_changed':
                # Future information must not leak into earlier tokens.
                np.testing.assert_allclose(actual['output'][:, :-1], baseline_output[:, :-1],
                                           rtol=1e-5, atol=1e-5)
            print(f'PASS {name}: {len(expected)} stages; output max error '
                  f'{np.max(np.abs(actual["output"] - expected["output"].numpy())):.3g}', flush=True)
    print('\nMaximum absolute errors across cases:')
    for stage, error in stage_max.items():
        print(f'  {stage:18s} {error:.3g}')


if __name__ == '__main__':
    main()
