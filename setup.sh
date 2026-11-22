#!/usr/bin/env bash
# Install this repository's pinned Python dependencies without training a model.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.venv}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in "$ENV_DIR/bin/python" python3.12 /opt/homebrew/bin/python3.12 \
      "$HOME/miniconda3/envs/cis5190/bin/python" python3; do
    if "$candidate" -c 'import sys; assert sys.version_info[:2] == (3,12)' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
case "${1:-}" in
  --help|-h)
    cat <<'HELP'
Usage: bash setup.sh [--check]

Creates or updates .venv using requirements.lock.txt, checks dependencies,
and installs a project-local Jupyter kernel. Python 3.12 is required.
No data collection, training, evaluation, Git push or model download is run.

Optional environment variables:
  PYTHON_BIN   Path to a Python 3.12 executable.
  ENV_DIR      Environment directory (default: project/.venv).

Example:
  PYTHON_BIN=/path/to/python3.12 bash setup.sh
  source .venv/bin/activate
  jupyter lab

The historical installer inside archive/legacy_support is preserved for reference.
Use this root installer for the current repository.
HELP
    exit 0 ;;
  --check|"") ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'; then
  echo "Set PYTHON_BIN to an installed Python 3.12 executable." >&2
  exit 1
fi
test -f "$PROJECT_DIR/requirements.lock.txt"
if [[ "${1:-}" == "--check" ]]; then
  echo "Python 3.12 and the dependency lock are available. Nothing was installed."
  exit 0
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.lock.txt"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m ipykernel install --prefix "$ENV_DIR" --name dragonthree --display-name 'DragonThree (Python 3.12)'
"$ENV_DIR/bin/python" -c 'import torch,numpy,scipy,sklearn,pandas,matplotlib,transformers,tokenizers,nbformat,nbclient; assert torch.__version__.split("+")[0] == "2.9.1"; print("All required imports passed; PyTorch:",torch.__version__)'
echo "Setup complete. Activate the environment with: source $ENV_DIR/bin/activate"
echo "Review the notebooks before running any budgeted experiment command."
