#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${FISHEYE_COMBINED_ROOT:-${REPO_DIR}/data/Custom/combined_caliper_fisheye3dod}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    DATASET_ROOT="$1"
    shift
fi

CONDA_ENV="${CONDA_ENV:-fisheye3d}"
GPUS="${GPUS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-29501}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/work_dirs/combined_fisheye_ocam}"
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_COMMAND="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_COMMAND="$(command -v conda)"
else
    CONDA_COMMAND="${HOME:?HOME is not set}/miniconda3/bin/conda"
fi
CONFIG="${REPO_DIR}/projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_combined_ocam.py"

[[ -f "${DATASET_ROOT}/caliper_annotations_ocam_800x800/fisheye3dod_infos_train.pkl" ]] || {
    printf 'Missing combined OCam dataset: %s\n' "${DATASET_ROOT}" >&2
    exit 1
}
[[ -x "${CONDA_COMMAND}" ]] || {
    printf 'Conda executable not found: %s\n' "${CONDA_COMMAND}" >&2
    exit 1
}

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export FISHEYE_COMBINED_ROOT="$(realpath "${DATASET_ROOT}")"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"${CONDA_COMMAND}" run --no-capture-output -n "${CONDA_ENV}" \
    python -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --nproc_per_node="${GPUS}" --master_port="${PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" "$@"
