import subprocess, os
from pathlib import Path

token = Path("kaggle_token.txt").read_text(encoding="utf-8").strip()
env = os.environ.copy()
env["KAGGLE_API_TOKEN"] = token

r = subprocess.run(
    ["kaggle", "kernels", "logs", "eugenfromkharkov/instanseg-kaggle-training"],
    env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print(r.stdout[-3000:] if r.stdout else "(no output)")
if r.stderr:
    print("STDERR:", r.stderr[-500:])
