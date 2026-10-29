# Models and checkpoints

Run all commands from the repository root. Python commands use the existing parent project environment (`../.venv`).

| Checkpoint | Parameters | Width | Heads | Layers | F32 weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M | 5,208,960 | 192 | 3 | 8 | ~21 MB |
| 28M | 27,821,184 | 384 | 6 | 12 | ~111 MB |

## Load either checkpoint

From this folder, use the existing parent Python environment for the one-time
download (about 21 MB for 5M or 111 MB for 28M). Optionally pass
`--revision <commit>` to pin it. The old `download_5m.py` command still defaults to 5M.

```sh
../.venv/bin/python scripts/download_model.py --model 5M
cargo run --features metal -- models/Timestral-5M patch-metal.npy

../.venv/bin/python scripts/download_model.py --model 28M
cargo run --features metal -- models/Timestral-28M patch-28m.npy
```

The Rust loader reads local files only. It checks the configuration, every tensor
name, shape and F32 dtype, and the expected parameter count for the selected size. All tensors
load onto the chosen device. It constructs the patch projection, learned forecast
query, complete transformer, and monotone quantile head.
The smoke check feeds deterministic `[1, 3, 64]` features through the real patch
projection and optionally saves its `[1, 3, width]` output as a NumPy file.
`loading::load_model` accepts either supported architecture; `loading::load_5m`
remains available and continues to reject larger models.

Compare that output with the original Python layer:

```sh
../.venv/bin/python scripts/compare_patch_embedding.py models/Timestral-5M patch-metal.npy
../.venv/bin/python scripts/compare_patch_embedding.py models/Timestral-28M patch-28m.npy
```

For CPU, omit `--features metal`. Without Python at runtime, you can also copy
`config.json` and `model.safetensors` from the official checkpoint into a local
directory and pass its path directly to the Rust executable.

The integration check uses seeded synthetic weights from the official Python
model for both sizes, verifies patch projection parity, and checks rejection of missing,
extra, wrong-shape, wrong-dtype tensors and unsupported architectures:

```sh
cargo build
../.venv/bin/python tests/check_loading.py target/debug/timestral-rs
```

Synthetic-checkpoint validation does not establish parity with released weights.

All checkpoint-driven diagnostic examples, comparison scripts and the native
benchmark accept either size. For example, after `cargo build --examples`:

```sh
../.venv/bin/python scripts/compare_network.py models/Timestral-28M
../.venv/bin/python scripts/compare_forecast.py models/Timestral-28M
```

The full 1,024-step single-block diagnostic bypasses the public forecast's output
caps and can generate enormous tail quantiles. For that case, the comparison checks
normalized network outputs end to end, then checks denormalization and advancing
refinement against Python using identical upstream inputs. This isolates nonlinear
amplification of F32 differences without widening tolerances. It still reports the
original end-to-end error. Complete public forecasts, including the 1,057-step
multi-block case, retain their end-to-end comparisons.

Use `scripts/benchmark.py --checkpoint models/Timestral-28M ...` to benchmark 28M;
the report records the selected model identity and configuration hash.
