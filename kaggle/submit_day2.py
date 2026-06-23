#!/usr/bin/env python3
"""Sync, push, wait, and submit one Day 2 competition slot (v16–v20)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
KERNEL_SLUG = "ilakkmanoharan/asra-security-submit"
COMPETITION = "ai-agent-security-multi-step-tool-attacks"

SLOTS: dict[int, dict[str, str]] = {
    1: {
        "mode": "blf",
        "label": "v16",
        "message": "ASRA-Security v16 - BLF - Refusal-aware belief revision",
    },
    2: {
        "mode": "blf",
        "label": "v17",
        "message": "ASRA-Security v17 - BLF - Predicate evidence accumulation",
    },
    3: {
        "mode": "harness",
        "label": "v18",
        "message": "ASRA-Security v18 - AutoHarness - Untrusted + privileged phases",
    },
    4: {
        "mode": "harness",
        "label": "v19",
        "message": "ASRA-Security v19 - AutoHarness - Dynamic branch_batch",
    },
    5: {
        "mode": "blf_harness",
        "label": "v20",
        "message": "ASRA-Security v20 - BLF+AutoHarness - Belief-guided phase transitions",
    },
}


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def kernel_status() -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if "COMPLETE" in output:
        return "COMPLETE"
    if "RUNNING" in output or "QUEUED" in output or "PENDING" in output:
        return "RUNNING"
    if "ERROR" in output:
        return "ERROR"
    return output.strip() or "UNKNOWN"


def wait_for_complete(timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = kernel_status()
        print(f"Kernel status: {status}")
        if status == "COMPLETE":
            return
        if status == "ERROR":
            raise RuntimeError("Kernel failed with ERROR status")
        time.sleep(30)
    raise TimeoutError(f"Kernel not complete after {timeout_s}s")


def push_and_get_version(python: str) -> int:
    result = subprocess.run(
        [python, str(KAGGLE_DIR / "push-submit-kernel.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    version = None
    for line in result.stdout.splitlines():
        if line.startswith("KERNEL_VERSION="):
            version = int(line.split("=", 1)[1])
    if version is None:
        raise RuntimeError("Could not parse KERNEL_VERSION from push output")
    return version


def submit_slot(slot: int, *, python: str, skip_wait: bool) -> None:
    if slot not in SLOTS:
        raise ValueError(f"Unknown slot {slot}")

    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("Set KAGGLE_API_TOKEN")

    spec = SLOTS[slot]
    run(
        [python, str(KAGGLE_DIR / "sync_submit_notebook.py"), "--mode", spec["mode"]],
        cwd=KAGGLE_DIR,
    )
    version = push_and_get_version(python)
    if not skip_wait:
        wait_for_complete()
    run(
        [
            "kaggle",
            "competitions",
            "submit",
            COMPETITION,
            "-f",
            "submission.csv",
            "-k",
            KERNEL_SLUG,
            "-m",
            spec["message"],
            "-v",
            str(version),
        ]
    )
    print(f"Submitted slot {slot} ({spec['label']}) as kernel version {version}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, choices=sorted(SLOTS))
    parser.add_argument("--all", action="store_true", help="Submit slots 1-5 sequentially")
    parser.add_argument("--from-slot", type=int, default=1)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    slots = (
        list(range(args.from_slot, 6))
        if args.all
        else [args.slot]
        if args.slot
        else []
    )
    if not slots:
        parser.error("Provide --slot N or --all")

    for slot in slots:
        submit_slot(slot, python=python, skip_wait=args.skip_wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
