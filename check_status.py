import subprocess, os, sys
from pathlib import Path

token = Path("kaggle_token.txt").read_text(encoding="utf-8").strip()
env = os.environ.copy()
env["KAGGLE_API_TOKEN"] = token

r = subprocess.run(
    ["kaggle", "kernels", "status", "eugenfromkharkov/instanseg-kaggle-training"],
    env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)
print("RC:", r.returncode)
