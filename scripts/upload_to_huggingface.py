#!/usr/bin/env python3
"""Upload LoRA adapter to HuggingFace Hub.

Usage:
    # First login:
    huggingface-cli login

    # Then upload:
    uv run python scripts/upload_to_huggingface.py \
        --checkpoint-dir checkpoints/v10_s40 \
        --repo-id omar81939/rlm-qwen35-35b-a3b \
        --model-card MODEL_CARD.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to HuggingFace")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with adapter files")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g. user/model-name)")
    parser.add_argument("--model-card", default=None, help="Path to MODEL_CARD.md (will be uploaded as README.md)")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    from huggingface_hub import HfApi, upload_folder

    api = HfApi()

    # Verify login
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    # Create repo if needed
    print(f"Creating/verifying repo: {args.repo_id}")
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)

    # Copy model card as README.md if provided
    ckpt_dir = Path(args.checkpoint_dir)
    if args.model_card:
        model_card_path = Path(args.model_card)
        if model_card_path.exists():
            readme_path = ckpt_dir / "README.md"
            readme_path.write_text(model_card_path.read_text())
            print(f"Copied {args.model_card} -> {readme_path}")

    # Upload
    print(f"Uploading {ckpt_dir} to {args.repo_id}...")
    upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=args.repo_id,
        commit_message="Upload RLM LoRA adapter (V17, RS-SFT trained, +21.7pp avg over base)",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
