"""Centralised download helpers for datasets not available in sklearn/torchvision."""

from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_local_path(candidates: list[str]) -> str | None:
    """Return the first existing path from *candidates*, or None."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def download_zip_member(
    url: str,
    dest_path: str,
    zip_member: str,
    timeout: int = 120,
) -> str:
    """Download a zip archive from *url* and extract a single *zip_member* to *dest_path*.

    If *dest_path* already exists the download is skipped (cache-hit).
    Returns *dest_path*.
    """
    if os.path.exists(dest_path):
        log.info("Cache hit: %s", dest_path)
        return dest_path
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", url)
    resp = urllib.request.urlopen(url, timeout=timeout)
    raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(zip_member) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
    log.info("Saved to %s", dest_path)
    return dest_path


def download_kaggle_dataset(
    dataset_id: str,
    dest_dir: str,
    kaggle_json: str | None = None,
) -> None:
    """Download a Kaggle dataset using the kaggle CLI.

    Requires ``kaggle`` to be installed and credentials either in
    *kaggle_json* (path to kaggle.json) or in ``~/.kaggle/kaggle.json``.

    Args:
        dataset_id: Kaggle dataset identifier, e.g. ``"puneet6060/intel-image-classification"``.
        dest_dir:   Directory where the dataset will be unzipped.
        kaggle_json: Optional path to kaggle.json credentials file.
    """
    import subprocess

    os.makedirs(dest_dir, exist_ok=True)
    env = os.environ.copy()
    if kaggle_json:
        env["KAGGLE_CONFIG_DIR"] = str(Path(kaggle_json).parent)

    log.info("Downloading Kaggle dataset %s into %s ...", dataset_id, dest_dir)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", dest_dir, "--unzip"],
        check=True,
        env=env,
    )
    log.info("Dataset %s downloaded.", dataset_id)
