#!/usr/bin/env python3
"""Push asra-security-submit.ipynb to Kaggle with T4 GPU (internet off)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

KAGGLE_DIR = Path(__file__).resolve().parent
META_PATH = KAGGLE_DIR / "kernel-metadata.json"
NOTEBOOK_PATH = KAGGLE_DIR / "asra-security-submit.ipynb"
PUSH_URL = "https://www.kaggle.com/api/v1/kernels/push"


def load_notebook() -> dict:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            if isinstance(cell.get("source"), list):
                cell["source"] = "".join(cell["source"])
    return nb


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("Set KAGGLE_API_TOKEN")
        return 1

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    nb = load_notebook()

    payload = {
        "slug": meta["id"],
        "newTitle": meta["title"],
        "text": json.dumps(nb),
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "notebook"),
        "isPrivate": meta.get("is_private", "true") == "true",
        "enableGpu": meta.get("enable_gpu", "true") == "true",
        "enableTpu": meta.get("enable_tpu", "false") == "true",
        "enableInternet": meta.get("enable_internet", "false") == "true",
        "competitionDataSources": meta.get("competition_sources", []),
        "machineShape": meta.get("machine_shape", "NvidiaTeslaT4"),
    }

    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(PUSH_URL, headers=headers, json=payload, timeout=120)
    print(f"HTTP {resp.status_code}")
    print(resp.text)
    if not resp.ok:
        return 1
    data = resp.json()
    version = data.get("versionNumber") or data.get("versionNumberNullable")
    if version is not None:
        print(f"KERNEL_VERSION={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
