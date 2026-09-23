#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL_ID = "eugenfromkharkov/instanseg-kaggle-training"
DATASET_SOURCE = "eugenfromkharkov/overfit-small-dataset"

def load_token(token_path="kaggle_token.txt") -> str:
    path = Path(token_path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle token file not found at: {path.resolve()}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Token file {path.resolve()} is empty.")
    return token

def get_kernel_status(env: dict) -> str:
    cmd = ["kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error checking kernel status:\n{res.stderr}", file=sys.stderr)
        return "UNKNOWN"
    return res.stdout.strip()


def ensure_dataset_source(metadata_path="kernel-metadata.json") -> None:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle kernel metadata not found at: {path.resolve()}")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    dataset_sources = metadata.setdefault("dataset_sources", [])
    if DATASET_SOURCE not in dataset_sources:
        dataset_sources.append(DATASET_SOURCE)
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

def fetch_and_display_logs(env: dict):
    print("\nFetching latest execution logs...", flush=True)
    cmd = ["kaggle", "kernels", "logs", KERNEL_ID]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0 and res.stdout:
        lines = res.stdout.splitlines()
        print("\n--- Execution Logs (Tail) ---", flush=True)
        for line in lines[-100:]:
            try:
                print(line)
            except Exception:
                print(line.encode("ascii", "replace").decode("ascii"))
        print("-----------------------------\n", flush=True)
    else:
        print("Could not fetch execution logs or logs were empty.", flush=True)

def run_and_monitor(poll_interval=60, timeout=1800, push=False):
    token = load_token()
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"

    if push:
        ensure_dataset_source()
        print(f"Triggering execution of kernel '{KERNEL_ID}' on P100 GPU...", flush=True)
        push_cmd = ["kaggle", "kernels", "push", "-p", ".", "--accelerator", "NvidiaTeslaP100"]
        res = subprocess.run(push_cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            print(f"Failed to start kernel execution:\n{res.stderr}", file=sys.stderr, flush=True)
            sys.exit(res.returncode)
        print(res.stdout, flush=True)
    else:
        print(f"Monitoring the latest execution of kernel '{KERNEL_ID}' without pushing...", flush=True)

    start_time = time.time()
    print("Monitoring execution status...", flush=True)

    while True:
        status_line = get_kernel_status(env)
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status_line}", flush=True)
        except Exception:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status_line.encode('ascii', 'replace').decode('ascii')}", flush=True)

        status_upper = status_line.upper()
        if "COMPLETE" in status_upper and "RUNNING" not in status_upper:
            print("\nKernel execution completed successfully!", flush=True)
            fetch_and_display_logs(env)
            break
        elif any(err in status_upper for err in ["ERROR", "CANCELLED", "FAILED"]):
            print(f"\nKernel execution failed with status: {status_line}", flush=True)
            fetch_and_display_logs(env)
            sys.exit(1)

        if time.time() - start_time > timeout:
            print(f"\nExecution timed out after {timeout} seconds.", flush=True)
            fetch_and_display_logs(env)
            sys.exit(1)

        time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(
        description="Monitor the latest Kaggle kernel execution, optionally pushing the notebook first."
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=True,
        help="Push and run the current notebook before monitoring it.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status and log checks (default: 60).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum monitoring time in seconds (default: 1800).",
    )
    args = parser.parse_args()
    run_and_monitor(
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        push=args.push,
    )


if __name__ == "__main__":
    main()
