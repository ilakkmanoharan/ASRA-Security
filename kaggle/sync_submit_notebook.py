#!/usr/bin/env python3
"""Embed attack.py into asra-security-submit.ipynb with the chosen SUBMISSION_MODE."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
ATTACK_PATH = KAGGLE_DIR.parent / "attack.py"
NOTEBOOK_PATH = KAGGLE_DIR / "asra-security-submit.ipynb"

WRITE_CELL = '''from pathlib import Path

attack_code = {attack_repr}

out = Path('/kaggle/working/attack.py')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(attack_code, encoding='utf-8')
print(f'attack.py written ({{out.stat().st_size}} bytes, mode={mode})')
'''


def load_attack(mode: str) -> str:
    text = ATTACK_PATH.read_text(encoding="utf-8")
    if not re.search(r'^SUBMISSION_MODE = "', text, re.MULTILINE):
        raise ValueError("attack.py missing SUBMISSION_MODE constant")
    return re.sub(
        r'^SUBMISSION_MODE = "[^"]*"',
        f'SUBMISSION_MODE = "{mode}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )


def sync_notebook(mode: str) -> None:
    attack = load_attack(mode)
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    new_source = WRITE_CELL.format(attack_repr=repr(attack), mode=mode)

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if "attack_code" in source and "/kaggle/working/attack.py" in source:
            cell["source"] = new_source
            cell["outputs"] = []
            cell["execution_count"] = None
            updated = True
            break

    if not updated:
        raise RuntimeError("Could not find attack.py write cell in submit notebook")

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    (KAGGLE_DIR / "attack.py").write_text(attack, encoding="utf-8")
    print(f"Synced notebook with SUBMISSION_MODE={mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["asra", "blf", "harness", "cwm", "asra_blf"],
    )
    args = parser.parse_args()
    sync_notebook(args.mode)


if __name__ == "__main__":
    main()
