#!/usr/bin/env bash
set -euo pipefail

readonly FLASHINFER_REPOSITORY="https://github.com/flashinfer-ai/flashinfer.git"
readonly FLASHINFER_COMMIT="56ed54098411946525da17eca2bcfd7ec72a515e"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PATCH="${ROOT}/third_party/flashinfer/aspire.patch"
readonly SOURCE="${FLASHINFER_SOURCE:-${ROOT}/third_party/flashinfer/source}"
readonly PYTHON="${PYTHON:-python}"
readonly EDITABLE="${FLASHINFER_EDITABLE:-0}"
readonly VERBOSE="${FLASHINFER_VERBOSE:-0}"

for value in EDITABLE VERBOSE; do
  if [[ "${!value}" != "0" && "${!value}" != "1" ]]; then
    echo "FLASHINFER_${value} must be 0 or 1" >&2
    exit 1
  fi
done

if [[ ! -d "${SOURCE}/.git" ]]; then
  if [[ -e "${SOURCE}" ]]; then
    echo "Refusing to replace non-Git path: ${SOURCE}" >&2
    exit 1
  fi
  git clone --no-checkout "${FLASHINFER_REPOSITORY}" "${SOURCE}"
  git -C "${SOURCE}" fetch origin "${FLASHINFER_COMMIT}"
  git -C "${SOURCE}" checkout --detach "${FLASHINFER_COMMIT}"
  git -C "${SOURCE}" submodule update --init --recursive
fi

actual_commit="$(git -C "${SOURCE}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${FLASHINFER_COMMIT}" ]]; then
  if [[ -n "$(git -C "${SOURCE}" status --porcelain)" ]]; then
    echo "FlashInfer source is dirty at the wrong commit: ${SOURCE}" >&2
    exit 1
  fi
  git -C "${SOURCE}" fetch origin "${FLASHINFER_COMMIT}"
  git -C "${SOURCE}" checkout --detach "${FLASHINFER_COMMIT}"
  git -C "${SOURCE}" submodule update --init --recursive
fi

if git -C "${SOURCE}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "ASPIRE FlashInfer patch already applied."
elif [[ -z "$(git -C "${SOURCE}" status --porcelain)" ]] \
  && git -C "${SOURCE}" apply --check "${PATCH}"; then
  git -C "${SOURCE}" apply "${PATCH}"
else
  echo "FlashInfer source has changes other than the ASPIRE patch: ${SOURCE}" >&2
  exit 1
fi

# Environments created by `uv venv` have no pip; use uv there.
if "${PYTHON}" -m pip --version >/dev/null 2>&1; then
  installer=("${PYTHON}" -m pip)
  install_args=(install --no-build-isolation)
elif command -v uv >/dev/null 2>&1; then
  installer=(uv pip)
  install_args=(install --python "${PYTHON}" --no-build-isolation)
else
  echo "Neither pip (in ${PYTHON}) nor uv is available" >&2
  exit 1
fi
if [[ "${VERBOSE}" == "1" ]]; then
  install_args+=(-v)
fi
if [[ "${EDITABLE}" == "1" ]]; then
  install_args+=(-e)
fi
"${installer[@]}" "${install_args[@]}" "${SOURCE}"
echo "Installed patched FlashInfer ${FLASHINFER_COMMIT} from ${SOURCE}"

FLASHINFER_VERSION="$("${PYTHON}" -c 'import flashinfer; print(flashinfer.__version__)')"
JIT_CACHE="${HOME}/.cache/flashinfer/${FLASHINFER_VERSION}"
if [[ -d "${JIT_CACHE}" ]]; then
  rm -rf "${JIT_CACHE}"
  echo "Cleared FlashInfer JIT cache: ${JIT_CACHE}"
fi
