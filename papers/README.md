# Vega paper run

Run everything from the root of your `MesoUQ` copy on Vega. Use `tmux` or `screen`.

## 1. Setup once

```bash
set -euo pipefail

test -f pyproject.toml
test -f papers/huq_emb/run_vega_50k_campaign.py

export REPO="$(pwd -P)"
export HPC_SITE=vega

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0 CMake/3.24.3-GCCcore-12.2.0 HDF5/1.14.0-gompi-2022b

python -m venv _vega/venv
source _vega/venv/bin/activate
python -m pip install --upgrade pip
# Vega's driver stack cannot run the default PyPI CUDA 13 PyTorch wheel.
# Install a CUDA 12.6 wheel first so the editable install keeps this build.
python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.11.0+cu126"
python -m pip install -e ".[test,mpi]"
python -m pip install pybind11 meson ninja h5py

bash scripts/platforms/hpc/bootstrap_korali.sh --jobs 8 --native-cuda-batch
source _vega/korali/env.sh
python - <<'PY'
import json
from pathlib import Path

build_options = Path("_vega/korali/build/meson-info/intro-buildoptions.json")
options = {item["name"]: item["value"] for item in json.loads(build_options.read_text())}
if options.get("native_cuda_batch") is not True:
    raise SystemExit("ERROR: Korali was not built with native_cuda_batch=true")
PY

# Mirheo's CMake configure step writes generated files into the Mirheo source
# tree. A fresh clone in another Vega account must therefore use a writable
# account-local Mirheo copy instead of the default lock path under another user.
export MESOUQ_MIRHEO_SRC="${MESOUQ_MIRHEO_SRC:-$HOME/software/Mirheo}"
if [[ ! -f "$MESOUQ_MIRHEO_SRC/CMakeLists.txt" ]]; then
  mkdir -p "$(dirname "$MESOUQ_MIRHEO_SRC")"
  rsync -rlt --chmod=u+rwX,go+rX /ceph/hpc/home/eubrieucb/software/Mirheo/ "$MESOUQ_MIRHEO_SRC/"
fi
bash scripts/platforms/hpc/bootstrap_mirheo.sh --source "$MESOUQ_MIRHEO_SRC" --jobs 8 --reconfigure
source _vega/mirheo/env.sh
bash scripts/platforms/vega/bootstrap_tex.sh
source _vega/tinytex/env.sh

python scripts/platforms/hpc/doctor_hpc.py --strict --with-mirheo --with-tex
python -m pytest tests/test_vega_50k_campaign.py tests/test_huq_emb_campaign_orchestrator.py tests/test_run_exact_uqdpd_asset_port.py

mkdir -p _vega/logs
sbatch --wait \
  --partition=dev \
  --gres=gpu:1 \
  --time=00:05:00 \
  --job-name=mesouq-torch-cuda-probe \
  --output="$REPO/_vega/logs/torch_cuda_probe_%j.out" \
  --error="$REPO/_vega/logs/torch_cuda_probe_%j.err" \
  --export=ALL,REPO="$REPO" <<'SBATCH'
#!/bin/bash
set -euo pipefail
cd "$REPO"
module purge
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.2.2
source _vega/venv/bin/activate
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
if torch.version.cuda != "12.6":
    raise SystemExit("ERROR: expected the CUDA 12.6 PyTorch wheel")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot initialize CUDA on this Vega GPU node")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
SBATCH
```

If Korali was already bootstrapped without `native_cuda_batch`, rerun the Korali
bootstrap with `--reconfigure --native-cuda-batch`. If PyTorch reports a CUDA 13
wheel, reinstall the CUDA 12.6 wheel above before running the paper campaign.

## 2. Run the four 50k campaigns

```bash
set -euo pipefail

export REPO="$(pwd -P)"
export HPC_SITE=vega
export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$HOME/mesouq_paper_data}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-huq_emb_50k_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$PAPER_DATA_ROOT/logs/$CAMPAIGN_ID"
printf '%s\n' "$CAMPAIGN_ID" > "$PAPER_DATA_ROOT/LAST_CAMPAIGN_ID.txt"

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0 HDF5/1.14.0-gompi-2022b
source _vega/venv/bin/activate
source _vega/korali/env.sh
source _vega/mirheo/env.sh
source _vega/tinytex/env.sh

python papers/huq_emb/run_vega_50k_campaign.py \
  --paper-data-root "$PAPER_DATA_ROOT" \
  --campaign-id "$CAMPAIGN_ID" \
  --python-bin "$(command -v python)" \
  --phase2-cpu-ranks 64 \
  --force-rebuild-assets
```

This runs:

- `compression:full-model:production`
- `compression:reduced-model:production`
- `indentation:full-model:production`
- `indentation:reduced-model:production`

Progress:

```bash
export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$HOME/mesouq_paper_data}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-$(cat "$PAPER_DATA_ROOT/LAST_CAMPAIGN_ID.txt")}"
squeue -u "$USER"
cat "$PAPER_DATA_ROOT/logs/$CAMPAIGN_ID/vega_50k_campaign_report.json"
```

## 3. Build the paper figures

Run this only after step 2 finished and the campaign report passed.

```bash
set -euo pipefail

export REPO="$(pwd -P)"
export HPC_SITE=vega
export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$HOME/mesouq_paper_data}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-$(cat "$PAPER_DATA_ROOT/LAST_CAMPAIGN_ID.txt")}"
mkdir -p "$PAPER_DATA_ROOT/logs/$CAMPAIGN_ID"

python - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["PAPER_DATA_ROOT"]) / "logs" / os.environ["CAMPAIGN_ID"] / "vega_50k_campaign_report.json"
report = json.loads(report_path.read_text())
failed_steps = [
    step for step in report.get("steps", [])
    if step.get("job_state") and step.get("job_state") != "COMPLETED"
]
if report.get("status") != "passed" or failed_steps:
    raise SystemExit(f"ERROR: 50k campaign has not passed: {report_path}")
PY

sbatch --wait \
  --partition=gpu \
  --gres=gpu:1 \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=64000 \
  --job-name=mesouq-paper-plots \
  --output="$PAPER_DATA_ROOT/logs/$CAMPAIGN_ID/exact_replay_%j.out" \
  --error="$PAPER_DATA_ROOT/logs/$CAMPAIGN_ID/exact_replay_%j.err" <<SBATCH
#!/bin/bash
set -euo pipefail
cd "$REPO"
export HPC_SITE=vega
module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0 HDF5/1.14.0-gompi-2022b
source _vega/venv/bin/activate
source _vega/korali/env.sh
source _vega/mirheo/env.sh
source _vega/tinytex/env.sh
python papers/huq_emb/run_exact_uqdpd_asset_port.py \
  --paper-data-root "$PAPER_DATA_ROOT" \
  --campaign-id "$CAMPAIGN_ID" \
  --python-bin "\$(command -v python)" \
  --site vega \
  --staging-device cuda \
  --texdeps-dir "$REPO/_vega/tinytex" \
  --force
SBATCH
```

## 4. Build the extra figures

Run this only after step 3 finished.

Mapping:

- `d1`: compression `2.1um`
- `d2`: compression `2.9um`
- `d3`: compression `3.0um`
- `d4`: indentation `3.2um`
- `d5`: indentation `3.4um`
- `d6`: indentation `5.8um`

```bash
set -euo pipefail

export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$HOME/mesouq_paper_data}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-$(cat "$PAPER_DATA_ROOT/LAST_CAMPAIGN_ID.txt")}"

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0 HDF5/1.14.0-gompi-2022b
source _vega/venv/bin/activate
source _vega/korali/env.sh
source _vega/mirheo/env.sh
source _vega/tinytex/env.sh

python papers/huq_emb/generate_out_of_scope_figures.py \
  --paper-data-root "$PAPER_DATA_ROOT" \
  --campaign-id "$CAMPAIGN_ID" \
  --max-posterior-samples 50000
```

Extra figures are in:

- `$PAPER_DATA_ROOT/figures/out_of_paper_scope/grouped_holdout_validation/`
- `$PAPER_DATA_ROOT/figures/out_of_paper_scope/sobol_sensitivity/`
- `$PAPER_DATA_ROOT/figures/out_of_paper_scope/posterior_marginals_phase1/`
- `$PAPER_DATA_ROOT/figures/out_of_paper_scope/map_vs_simulation/`

## 5. Check

```bash
set -euo pipefail

export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$HOME/mesouq_paper_data}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-$(cat "$PAPER_DATA_ROOT/LAST_CAMPAIGN_ID.txt")}"

module purge
module load Python/3.10.8-GCCcore-12.2.0
source _vega/venv/bin/activate

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER_DATA_ROOT"])
cid = os.environ["CAMPAIGN_ID"]

checks = [
    ("50k campaign", root / "logs" / cid / "vega_50k_campaign_report.json", "status", "passed"),
    ("paper release", root / "manifests" / "paper_release_manifest.json", "release_status", "PASS"),
    ("paper figures", root / "runs" / cid / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json", "status", "passed"),
    ("extra figures", root / "figures" / "out_of_paper_scope" / "out_of_scope_figures_manifest.json", "status", "passed"),
]

for label, path, key, expected in checks:
    payload = json.loads(path.read_text())
    actual = payload.get(key)
    print(f"{label}: {actual}")
    if actual != expected:
        raise SystemExit(f"ERROR: {label} failed: {path}")

extra = json.loads((root / "figures" / "out_of_paper_scope" / "out_of_scope_figures_manifest.json").read_text())
warnings = extra.get("data_quality_warnings", [])
if warnings:
    print("extra figure warnings:")
    for warning in warnings:
        print(f"  - {warning}")
    raise SystemExit("ERROR: rerun from the full 50k campaign outputs.")

print("main:", root / "figures" / "main")
print("supp:", root / "figures" / "supplementary")
print("extra:", root / "figures" / "out_of_paper_scope")
print("tables:", root / "tables")
PY
```

If it fails:

```bash
find "$PAPER_DATA_ROOT/logs/$CAMPAIGN_ID" \( -name '*.stderr.log' -o -name '*.err' \) -type f -size +0 -print
```
