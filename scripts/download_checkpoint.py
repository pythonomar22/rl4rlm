#!/usr/bin/env python3
"""Download a Tinker checkpoint for HuggingFace upload.

Usage:
    uv run python scripts/download_checkpoint.py \
        --session bae6fabb-fcd7-58a6-b0af-101131f8e6a6 \
        --state state-0005 \
        --output checkpoints/v11_s5.tar.gz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import tinker


def main():
    parser = argparse.ArgumentParser(description="Download Tinker checkpoint")
    parser.add_argument("--session", required=True, help="Tinker session ID")
    parser.add_argument("--state", required=True, help="Weights checkpoint name (e.g. checkpoint-0005)")
    parser.add_argument("--output", required=True, help="Output file path (.tar.gz)")
    args = parser.parse_args()

    tinker_path = f"tinker://{args.session}:train:0/weights/{args.state}"
    print(f"Downloading checkpoint from: {tinker_path}")

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    print("Getting download URL...")
    url = rest_client.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    result = url.result()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {output_path}")
    with open(output_path, "wb") as f:
        f.write(result)

    print(f"Done! Checkpoint saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
