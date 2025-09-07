
from subprocess import run
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
S = BASE / "src"
for step in ["data_core_generate.py","models_core.py"]:
    print(f"---> Running {step}")
    run([sys.executable, str(S / step)], check=True)
print("All core results generated under results/.")
