"""Download an official 5M or 28M checkpoint into the ignored models/ folder."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["5M", "28M"], default="5M")
    parser.add_argument("--revision", default="main", help="Hub revision; use a commit hash to pin weights")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = f"Timestral-{args.model}"
    path = snapshot_download(
        repo_id=f"mistralai/{name}",
        revision=args.revision,
        local_dir=root / "models" / name,
        cache_dir=root.parent / ".cache" / "huggingface" / "hub",
        allow_patterns=["config.json", "model.safetensors", "LICENSE"],
    )
    print(f"Downloaded {args.model} checkpoint to {path}")


if __name__ == "__main__":
    main()
