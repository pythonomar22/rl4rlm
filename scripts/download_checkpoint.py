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
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import tinker


def main():
    parser = argparse.ArgumentParser(description="Download Tinker checkpoint")
    parser.add_argument("--session", required=True, help="Tinker session ID")
    parser.add_argument("--state", required=True, help="State checkpoint name (e.g. state-0005)")
    parser.add_argument("--output", required=True, help="Output file path (.tar.gz)")
    parser.add_argument("--name", default=None, help="Name for sampler weights (default: download-{state})")
    args = parser.parse_args()

    state_path = f"tinker://{args.session}:train:0/weights/{args.state}"
    sampler_name = args.name or f"download-{args.state}"

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    # Step 1: Load training state
    print(f"Loading training state: {state_path}")
    training_client = service_client.create_training_client_from_state(state_path)

    # Step 2: Create sampler weights (downloadable format)
    print(f"Creating sampler weights '{sampler_name}'...")
    result = training_client.save_weights_for_sampler(name=sampler_name).result()
    sampler_path = result.path
    print(f"Sampler path: {sampler_path}")

    # Step 3: Get signed download URL
    print("Getting download URL...")
    url_response = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()

    # Step 4: Download
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to: {output_path}")
    urllib.request.urlretrieve(url_response.url, str(output_path))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Done! Saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
