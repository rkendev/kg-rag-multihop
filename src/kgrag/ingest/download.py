"""Download 2WikiMultiHopQA (Apache-2.0) and stage the dev split.

Source: Alab-NII/2wikimultihop official `data.zip` (Dropbox direct-download form).
The raw archive and its contents land under ``data/raw/`` (gitignored). The dataset's
Apache-2.0 LICENSE is fetched separately and retained at ``data/LICENSE`` (committed),
and we re-verify the license header before declaring success.
"""
from __future__ import annotations

import sys
import zipfile

import requests

from .. import config


def _download(url: str, dest, *, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _verify_apache_license(path) -> None:
    head = path.read_text(encoding="utf-8", errors="replace")[:600]
    if "Apache License" not in head or "Version 2.0" not in head:
        raise SystemExit(
            f"LICENSE at {path} is not the expected Apache-2.0 text; refusing to proceed."
        )


def main() -> int:
    config.RAW.mkdir(parents=True, exist_ok=True)

    # 1. License first — gate everything else on it.
    license_dest = config.DATA / "LICENSE"
    print(f"fetching dataset LICENSE -> {license_dest}")
    _download(config.LICENSE_URL, license_dest)
    _verify_apache_license(license_dest)
    print("license verified: Apache-2.0")

    # 2. Dataset archive.
    zip_dest = config.RAW / "data.zip"
    if config.DEV_JSON.exists():
        print(f"dev split already present at {config.DEV_JSON}; skipping download")
        return 0
    print(f"downloading data.zip -> {zip_dest}")
    _download(config.DATA_ZIP_URL, zip_dest)
    print(f"downloaded {zip_dest.stat().st_size / 1e6:.1f} MB")

    # 3. Extract into data/raw/.
    with zipfile.ZipFile(zip_dest) as zf:
        names = zf.namelist()
        zf.extractall(config.RAW)
    print(f"extracted {len(names)} entries")

    # The archive may nest files; locate dev.json and normalise its location.
    if not config.DEV_JSON.exists():
        candidates = list(config.RAW.rglob("dev.json"))
        if not candidates:
            print(f"ERROR: dev.json not found after extraction. entries: {names[:20]}")
            return 1
        src = candidates[0]
        if src != config.DEV_JSON:
            src.replace(config.DEV_JSON)
    print(f"dev split ready at {config.DEV_JSON} "
          f"({config.DEV_JSON.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
