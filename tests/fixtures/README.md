These are the official checkpoint configurations used for architecture validation
and synthetic-weight loader tests. No model weights are committed.

- `config-5m.json`: `mistralai/Timestral-5M`, revision `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`.
- `config-28m.json`: `mistralai/Timestral-28M`, revision `49b26409ac87b5935539a5d8ffe7f397b720db47`.

Categorical preprocessing fixtures use the official Timestral reference revision
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`:

- `categorical-encoding.json`: 23 encoding cases and 28 exact target-sum cases.
- `categorical-known-normalization.json`: 22 constant/near-constant normalization cases.

From this model repository, regenerate using the pinned CPU Docker environment:

```sh
docker run --rm --platform linux/arm64 -v "$PWD:/work" timestral-python \
  python3 scripts/generate_categorical_fixtures.py
cargo test --locked
```

The fixtures record reference source hashes and numerical environment information.
No server checkout or checkpoint weights are needed to regenerate them.
