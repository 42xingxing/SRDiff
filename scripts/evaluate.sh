#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${WORK_DIR}"

usage() {
    echo "Usage: bash scripts/evaluate.sh <base|cmca> <GPU_LIST> <NUM_GPUS> --ckpt CKPT [--ensemble N] [--inference-steps N] [--pools 1,2,4]"
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

CKPT=""
ENSEMBLE=1
INFERENCE_STEPS=10
POOLS="1,2,4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            [[ $# -ge 2 ]] || { echo "--ckpt requires a checkpoint file"; exit 2; }
            CKPT=$2
            shift 2
            ;;
        --ensemble)
            [[ $# -ge 2 ]] || { echo "--ensemble requires an integer"; exit 2; }
            ENSEMBLE=$2
            shift 2
            ;;
        --inference-steps)
            [[ $# -ge 2 ]] || { echo "--inference-steps requires an integer"; exit 2; }
            INFERENCE_STEPS=$2
            shift 2
            ;;
        --pools)
            [[ $# -ge 2 ]] || { echo "--pools requires a comma-separated list"; exit 2; }
            POOLS=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

[[ -n "${CKPT}" ]] || { echo "--ckpt is required"; usage; exit 2; }
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}"; exit 2; }
[[ "${CKPT}" == *.ckpt ]] || { echo "--ckpt expects a Lightning .ckpt file"; exit 2; }
[[ "${ENSEMBLE}" =~ ^[1-9][0-9]*$ ]] || { echo "--ensemble must be a positive integer"; exit 2; }
[[ "${INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "--inference-steps must be a positive integer"; exit 2; }

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
IFS=',' read -r -a POOL_LIST <<< "${POOLS}"
for POOL in "${POOL_LIST[@]}"; do
    case "${POOL}" in
        1|2|4) ;;
        *) echo "Unsupported pool size: ${POOL} (expected 1, 2, or 4)"; exit 2 ;;
    esac

    "${PYTHON}" -m srdiff.cli \
        --eval \
        --gpus "${NUM_GPUS}" \
        --config-path "${CONFIG}" \
        --resume-run "${CKPT}" \
        --ensemble-times "${ENSEMBLE}" \
        --inference-steps "${INFERENCE_STEPS}" \
        --eval-postfix "_pool${POOL}"
done
