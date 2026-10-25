"""Compare transformer + quantile head on explicit normalized-network inputs."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.model import TimestralModel

from compare_quantile_head import reference, assert_ordered, run_binary
from compare_transformer import cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/examples/network'))
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    binary = args.binary.resolve()
    model = TimestralModel(TimestralConfig.load(checkpoint / 'config.json')).eval()
    model.load_state_dict(load_file(checkpoint / 'model.safetensors'), strict=True)
    with torch.inference_mode(), tempfile.TemporaryDirectory(prefix='timestral-network-') as tmp:
        inputs, output = Path(tmp) / 'inputs.safetensors', Path(tmp) / 'trace.safetensors'
        for name, (x, positions, time, variable, split) in cases(model):
            hidden = model.transformer(x, time, variable, positions)
            decoder = hidden[:, split:]
            expected = reference(model, tokens=decoder)
            expected.update(hidden=hidden, decoder=decoder)
            official = model._run_two_stream(x[:, :split], x[:, split:], time, variable, positions)
            torch.testing.assert_close(expected['quantiles'], official, rtol=1e-4, atol=2e-4)
            values = dict(x=x.numpy(), positions=positions.numpy().astype(np.float32),
                          time_allowed=time.numpy().astype(np.uint8), encoder_patches=np.array([split], dtype=np.uint32))
            if variable is not None:
                values['variable_allowed'] = variable.numpy().astype(np.uint8)
            actual = run_binary(binary, checkpoint, values, inputs, output)
            assert set(actual) == set(expected), 'Missing or extra trace stage'
            for stage, want in expected.items():
                np.testing.assert_allclose(actual[stage], want.numpy(), rtol=1e-4, atol=2e-4,
                                           err_msg=f'{name}/{stage}')
            assert_ordered(actual, model.config.median_index)
            np.testing.assert_array_equal(actual['quantiles'], actual['direct_quantiles'])
            error = np.max(np.abs(actual['quantiles'] - expected['quantiles'].numpy()))
            print(f'PASS {name}: normalized quantiles {actual["quantiles"].shape}; max error {error:.3g}', flush=True)


if __name__ == '__main__':
    main()
