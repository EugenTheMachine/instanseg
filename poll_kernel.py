"""Poll kernel status and last 50 log lines, write to result.txt."""
import subprocess, os
from pathlib import Path

token = Path("kaggle_token.txt").read_text(encoding="utf-8").strip()
env = os.environ.copy()
env["KAGGLE_API_TOKEN"] = token

KERNEL = "eugenfromkharkov/instanseg-kaggle-training"

status = subprocess.run(
    ["kaggle", "kernels", "status", KERNEL],
    env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
).stdout.strip()

logs_r = subprocess.run(
    ["kaggle", "kernels", "logs", KERNEL],
    env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
)
log_tail = "\n".join(logs_r.stdout.splitlines()[-50:]) if logs_r.stdout else "(no logs yet)"

out = f"STATUS: {status}\n\nLOG TAIL:\n{log_tail}\n"
Path("poll_result.txt").write_text(out, encoding="utf-8")
print(out)
