#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_DIR="${ROOT_DIR}/external_baselines"

mkdir -p "${BASELINE_DIR}"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dest="${BASELINE_DIR}/${name}"

  if [[ -d "${dest}/.git" ]]; then
    echo "Updating ${name}"
    git -C "${dest}" pull --ff-only
  else
    echo "Cloning ${name}"
    git clone "${url}" "${dest}"
  fi
}

clone_or_update "LLMLingua" "https://github.com/microsoft/LLMLingua.git"
clone_or_update "adaptive-k-retrieval" "https://github.com/megagonlabs/adaptive-k-retrieval.git"
clone_or_update "FLARE" "https://github.com/jzbjyb/FLARE.git"

cat <<'MSG'

External baseline code is now under:
  external_baselines/LLMLingua
  external_baselines/adaptive-k-retrieval
  external_baselines/FLARE
MSG
