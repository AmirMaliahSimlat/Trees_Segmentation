"""Load Hugging Face token from environment / .env for authenticated Hub access."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_files() -> None:
    """Load repo-root ``.env`` if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def ensure_hf_auth(*, verbose: bool = True) -> bool:
    """
    Authenticate to the Hugging Face Hub when ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` is set.

    Returns True if a token was applied.
    """
    load_dotenv_files()
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        if verbose:
            print(
                "No HF_TOKEN found. Create one at https://huggingface.co/settings/tokens "
                "and put HF_TOKEN=... in a project .env file (see .env.example)."
            )
        return False

    # Ensure both common env names are set for huggingface_hub / transformers
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"HF login warning: {exc}")
        # Token in env is still enough for most downloads
    else:
        if verbose:
            print("Authenticated with Hugging Face Hub (HF_TOKEN).")
    return True
