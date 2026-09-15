# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Derived from gpt-fast and distributed under the BSD 3-Clause License.
# See LICENSES/BSD-3-Clause-gpt-fast.txt.

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--hf_token", default=None, help="token for gated models")
    parser.add_argument("--out_dir", type=Path, required=True, help="local checkpoint directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo_id, local_dir=args.out_dir, token=args.hf_token)


if __name__ == "__main__":
    main()
