"""Compare complete Python/PyTorch and Rust/Candle forecasts in persistent processes.

Each timed forecast begins with CPU history and ends with a copied CPU F32 result.
Model loading, CSV/JSON/NumPy I/O, and correctness checks are outside forecast timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

DEFAULT_OPTIONS = dict(max_context=None, per_shot_horizon_patches=32,
                       quantile_real_cap_k=10000.0, extreme_value_threshold=1e30,
                       symmetric_averaging=True, clip_nonnegative=True)


def summarize(result):
    samples = np.asarray(result['warm_samples_ms'], dtype=np.float64)
    result.update(warm_median_ms=float(np.median(samples)), warm_p90_ms=float(np.percentile(samples, 90)),
                  warm_min_ms=float(samples.min()), warm_max_ms=float(samples.max()))
    return result


def python_benchmark(checkpoint, history, horizon, options, device, warmups, iterations, threads):
    import torch
    from timestral import TimestralForecaster

    if device == 'mps' and not torch.backends.mps.is_available():
        return dict(engine='python', device=device, status='unavailable', reason='PyTorch MPS is unavailable'), None
    torch.set_num_threads(threads)

    def synchronize():
        if device == 'mps':
            torch.mps.synchronize()

    started = time.perf_counter()
    synchronize()
    device_init_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    model = TimestralForecaster.from_pretrained(str(checkpoint), device=device,
        max_context=options['max_context'], per_shot_horizon_patches=options['per_shot_horizon_patches'],
        quantile_real_cap_k=options['quantile_real_cap_k'])
    model.extreme_value_threshold = options['extreme_value_threshold']
    synchronize()
    load_ms = (time.perf_counter() - started) * 1000

    def predict():
        synchronize()
        started = time.perf_counter()
        forecast = model.forecast(history, prediction_length=horizon,
            symmetric_averaging=options['symmetric_averaging'], clip_nonnegative=options['clip_nonnegative'])
        values = forecast.quantiles.cpu().numpy().copy().reshape(horizon, 21)
        synchronize()
        return (time.perf_counter() - started) * 1000, values

    with torch.inference_mode():
        first_ms, values = predict()
        assert np.isfinite(values).all()
        for _ in range(warmups):
            predict()
        samples = []
        for _ in range(iterations):
            ms, values = predict()
            samples.append(ms)
    return summarize(dict(engine='python', device=device, status='ok', load_ms=load_ms,
        device_init_ms=device_init_ms,
        first_forecast_ms=first_ms, warm_samples_ms=samples, torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(), options=options)), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/Timestral-5M'))
    parser.add_argument('--history', type=Path, default=Path('examples/history.csv'))
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rust-binary', type=Path, default=Path('target/release/examples/benchmark'))
    parser.add_argument('--rust-devices', nargs='+', choices=['cpu', 'metal'], default=['cpu', 'metal'])
    parser.add_argument('--python-devices', nargs='+', choices=['cpu', 'mps'], default=['cpu', 'mps'])
    parser.add_argument('--options', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/benchmark.json'))
    args = parser.parse_args()
    if min(args.horizon, args.iterations, args.threads) < 1 or args.warmups < 0:
        parser.error('Horizon, iterations and threads must be positive; warmups must be nonnegative')
    binary = args.rust_binary.resolve()
    if not binary.is_file():
        parser.error('Build first: cargo build --release --features metal --example benchmark')
    # Set before importing torch / creating Rust worker pools.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'CANDLE_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    checkpoint, history_path = args.checkpoint.resolve(), args.history.resolve()
    config = json.loads((checkpoint / 'config.json').read_text())
    if (config['d_model'], len(config['block_schedule'])) not in ((192, 8), (384, 12)):
        parser.error('Expected Timestral-5M or Timestral-28M')
    model_name = 'Timestral-5M' if config['d_model'] == 192 else 'Timestral-28M'
    history = np.loadtxt(history_path, dtype=np.float32, ndmin=1)
    if history.ndim != 1 or not len(history):
        parser.error('Expected one numeric history value per CSV line')
    options = DEFAULT_OPTIONS.copy()
    if args.options:
        override = json.loads(args.options.read_text())
        if not isinstance(override, dict) or override.keys() - options.keys():
            parser.error('Unknown forecast option')
        options.update(override)
    results, outputs = [], {}
    print(f'{model_name} | {len(history)} history -> {args.horizon} forecast steps | '
          f'F32 | CPU threads {args.threads}', flush=True)
    print(f'First forecast, then {args.warmups} warmups and {args.iterations} measured forecasts; '
          f'symmetric averaging={options["symmetric_averaging"]}, clipping={options["clip_nonnegative"]}.', flush=True)
    with tempfile.TemporaryDirectory(prefix='timestral-benchmark-') as tmp:
        settings = Path(tmp) / 'options.json'
        settings.write_text(json.dumps(options))
        for device in args.rust_devices:
            print(f'Benchmarking Rust/{device}...', flush=True)
            output = Path(tmp) / f'rust-{device}.npy'
            run = subprocess.run([str(binary), str(checkpoint), str(history_path), str(args.horizon),
                                  str(args.warmups), str(args.iterations), device, str(output), str(settings)],
                                 text=True, capture_output=True)
            if run.returncode:
                result = dict(engine='rust', device=device, status='error', reason=run.stderr.strip())
            else:
                result = summarize(json.loads(run.stdout))
                outputs[('rust', device)] = np.load(output, allow_pickle=False)
            results.append(result)
        for device in args.python_devices:
            print(f'Benchmarking Python/{device}...', flush=True)
            try:
                result, values = python_benchmark(checkpoint, history, args.horizon, options, device,
                                                  args.warmups, args.iterations, args.threads)
                if values is not None:
                    outputs[('python', device)] = values
            except Exception as error:
                detail = f'{type(error).__name__}: {error}'
                if device == 'mps' and isinstance(error, TypeError) and 'float64' in str(error) and 'MPS' in str(error):
                    result = dict(engine='python', device=device, status='unsupported',
                                  reason='Official Timestral uses float64 statistics; PyTorch MPS does not support them.',
                                  error_detail=detail)
                else:
                    result = dict(engine='python', device=device, status='error', reason=detail)
            results.append(result)

    reference = outputs.get(('python', 'cpu'))
    baseline = next((r['warm_median_ms'] for r in results
                     if r['engine'] == 'python' and r['device'] == 'cpu' and r['status'] == 'ok'), None)
    finite = history[np.isfinite(history)]
    scale = max(1., float(np.std(finite.astype(np.float64)))) if finite.size else 1.
    parity_failed = False
    for result in results:
        if result['status'] != 'ok':
            continue
        if baseline is not None:
            result['speedup_vs_python_cpu'] = baseline / result['warm_median_ms']
        if reference is not None:
            actual = outputs[(result['engine'], result['device'])]
            result['max_abs_error_vs_python_cpu'] = float(np.max(np.abs(actual - reference)))
            try:
                np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=2e-4 * scale)
                result['parity_passed'] = True
            except AssertionError:
                result['parity_passed'] = False
                parity_failed = True
    print('\nImplementation   Load ms   First ms   Warm median ms   Warm p90 ms   Speedup vs Python CPU')
    for result in results:
        name = f"{result['engine']}/{result['device']}"
        if result['status'] == 'ok':
            speedup = f"{result['speedup_vs_python_cpu']:.2f}x" if baseline is not None else 'n/a'
            print(f"{name:16s} {result['load_ms']:8.2f} {result['first_forecast_ms']:10.2f} "
                  f"{result['warm_median_ms']:16.2f} {result['warm_p90_ms']:13.2f} {speedup:>23s}")
        else:
            print(f"{name:16s} {result['status']}: {result['reason'].splitlines()[0]}")
    compared = [r for r in results if r['status'] == 'ok' and (r['engine'], r['device']) != ('python', 'cpu')]
    if reference is not None and compared:
        print('Output parity:', 'FAILED' if parity_failed else 'passed for all successful implementations')
    elif reference is not None:
        print('Output parity was not checked: no other implementation succeeded.')
    else:
        print('Python CPU was not benchmarked successfully; output parity was not checked.')
    report = dict(workload=dict(model=f'mistralai/{model_name}', checkpoint=str(checkpoint),
        config_sha256=hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest(),
        history=str(history_path), history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
        history_steps=len(history), horizon=args.horizon, dtype='float32', options=options),
        measurement=dict(warmups=args.warmups, iterations=args.iterations, cpu_threads=args.threads,
                         synchronized=True, includes_cpu_readback=True, platform=platform.platform(),
                         python=sys.version, mps_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')),
        results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output}')
    required_cpu_failed = any(r['device'] == 'cpu' and r['status'] != 'ok' for r in results)
    if parity_failed or required_cpu_failed or not outputs:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
