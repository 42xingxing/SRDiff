#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${WORK_DIR}"

usage() {
    echo "Usage: bash scripts/train.sh <base|cmca> <GPU_LIST> <NUM_GPUS> [--resume CKPT] [--debug] [--add-datetime-prefix]"
}

if [[ $# -lt 3 ]]; then
    usage
    exit 2
fi

VARIANT=$1
GPU_LIST=$2
NUM_GPUS=$3
shift 3

[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_GPUS must be a positive integer"; exit 2; }
validate_gpu_selection "${GPU_LIST}" "${NUM_GPUS}"

case "${VARIANT}" in
    base|cmca) CONFIG="configs/srdiff/${VARIANT}.yaml" ;;
    *) echo "Unknown variant: ${VARIANT}"; usage; exit 2 ;;
esac

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            [[ $# -ge 2 ]] || { echo "--resume requires a checkpoint file"; exit 2; }
            [[ -f "$2" ]] || { echo "Checkpoint not found: $2"; exit 2; }
            [[ "$2" == *.ckpt ]] || { echo "--resume expects a Lightning .ckpt file"; exit 2; }
            EXTRA_ARGS+=(--resume-run "$2")
            shift 2
            ;;
        --debug)
            EXTRA_ARGS+=(--debug)
            shift
            ;;
        --add-datetime-prefix)
            EXTRA_ARGS+=(--add-datetime-prefix)
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
"${PYTHON}" -m srdiff.cli \
    --train \
    --gpus "${NUM_GPUS}" \
    --config-path "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
