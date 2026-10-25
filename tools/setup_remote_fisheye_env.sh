#!/usr/bin/env bash
# Bootstrap Miniconda and prepare a Python 3.8 MMDetection3D/Fisheye3DOD env.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_DIR="${HOME:?HOME is not set}/miniconda3"
ENV_NAME="fisheye3d"
SOURCE_DIR=""
CUDA_TAG="auto"
MMDET3D_URL="https://github.com/open-mmlab/mmdetection3d.git"
FISHEYE_URL="https://github.com/weiyangdaren/Fisheye3DOD.git"
INSTALL_OCAM=1
EXTRA_REQUIREMENTS=()
LOCAL_PACKAGES=()

usage() {
    cat <<'EOF'
Usage: tools/setup_remote_fisheye_env.sh [options]

Installs Miniconda when conda is unavailable, creates a Python 3.8 conda
environment, obtains MMDetection3D v1.4.0 and Fisheye3DOD, initializes all
submodules, and installs the complete training/runtime dependency stack.
It does not start training.

Options:
  --conda-dir PATH            Miniconda installation (default: ~/miniconda3)
  --env-name NAME            Conda environment name (default: fisheye3d)
  --source-dir PATH           MMDetection3D checkout destination
  --cuda auto|cu118|cu121     PyTorch CUDA build (default: detect from nvcc)
  --extra-requirements PATH   Additional compatible requirements; repeatable
  --local-package PATH        Additional compatible editable repo; repeatable
  --no-ocam                   Skip ocamcalib_undistort
  -h, --help                  Show this help

Example:
  ./tools/setup_remote_fisheye_env.sh \
    --conda-dir /opt/user/miniconda3 \
    --env-name fisheye3d \
    --source-dir /opt/work/mmdetection3d \
    --cuda cu118

After setup:
  source /opt/user/miniconda3/etc/profile.d/conda.sh
  conda activate fisheye3d
  cd /opt/work/mmdetection3d
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

abspath() {
    realpath -m "$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-dir) CONDA_DIR="${2:?missing value for --conda-dir}"; shift 2 ;;
        --env-name) ENV_NAME="${2:?missing value for --env-name}"; shift 2 ;;
        --source-dir) SOURCE_DIR="${2:?missing value for --source-dir}"; shift 2 ;;
        --cuda) CUDA_TAG="${2:?missing value for --cuda}"; shift 2 ;;
        --extra-requirements) EXTRA_REQUIREMENTS+=("${2:?missing value for --extra-requirements}"); shift 2 ;;
        --local-package) LOCAL_PACKAGES+=("${2:?missing value for --local-package}"); shift 2 ;;
        --no-ocam) INSTALL_OCAM=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

CONDA_DIR="$(abspath "${CONDA_DIR}")"
if [[ -z "${SOURCE_DIR}" ]]; then
    if [[ -f "${LOCAL_REPO_DIR}/mmdet3d/version.py" ]]; then
        SOURCE_DIR="${LOCAL_REPO_DIR}"
    else
        SOURCE_DIR="${PWD}/mmdetection3d"
    fi
fi
SOURCE_DIR="$(abspath "${SOURCE_DIR}")"

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "Invalid conda environment name: ${ENV_NAME}"

install_miniconda() {
    if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
        CONDA_EXE_PATH="${CONDA_DIR}/bin/conda"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_EXE_PATH="$(command -v conda)"
        printf 'Using existing conda: %s\n' "${CONDA_EXE_PATH}"
        return
    fi

    local architecture installer_url installer
    architecture="$(uname -m)"
    case "${architecture}" in
        x86_64) architecture="x86_64" ;;
        aarch64|arm64) architecture="aarch64" ;;
        *) die "Unsupported CPU architecture for Miniconda: ${architecture}" ;;
    esac

    installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${architecture}.sh"
    installer="$(mktemp "${TMPDIR:-/tmp}/miniconda.XXXXXX.sh")"

    printf 'Downloading Miniconda from %s\n' "${installer_url}"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "${installer}" "${installer_url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="${installer}" "${installer_url}"
    else
        die "curl or wget is required to download Miniconda"
    fi

    [[ ! -e "${CONDA_DIR}" ]] || \
        die "Conda destination exists but has no conda executable: ${CONDA_DIR}"
    mkdir -p "$(dirname "${CONDA_DIR}")"
    bash "${installer}" -b -p "${CONDA_DIR}"
    rm -f -- "${installer}"
    CONDA_EXE_PATH="${CONDA_DIR}/bin/conda"
    "${CONDA_EXE_PATH}" config --set auto_activate_base false
}

choose_cuda() {
    if [[ "${CUDA_TAG}" != "auto" ]]; then
        [[ "${CUDA_TAG}" == "cu118" || "${CUDA_TAG}" == "cu121" ]] || \
            die "--cuda must be auto, cu118, or cu121"
        return
    fi

    command -v nvcc >/dev/null 2>&1 || \
        die "nvcc is required for Fisheye3DOD CUDA ops; install CUDA 11.8/12.1 or add nvcc to PATH"
    local major
    major="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -1)"
    case "${major}" in
        11) CUDA_TAG="cu118" ;;
        12) CUDA_TAG="cu121" ;;
        *) die "Unsupported CUDA compiler; CUDA 11.8 or 12.1 is required" ;;
    esac
}

create_environment() {
    if "${CONDA_EXE_PATH}" run -n "${ENV_NAME}" \
        python -c 'import sys' >/dev/null 2>&1; then
        printf 'Reusing conda environment: %s\n' "${ENV_NAME}"
    else
        "${CONDA_EXE_PATH}" create --yes --override-channels --channel conda-forge \
            --name "${ENV_NAME}" \
            python=3.8 pip=24.0 'setuptools<70' wheel ninja cmake
    fi

    local version
    version="$("${CONDA_EXE_PATH}" run -n "${ENV_NAME}" python -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${version}" == "3.8" ]] || \
        die "Existing environment ${ENV_NAME} uses Python ${version}, not Python 3.8"
}

obtain_sources() {
    command -v git >/dev/null 2>&1 || die "git is required"

    if [[ -d "${SOURCE_DIR}/.git" ]]; then
        [[ -f "${SOURCE_DIR}/mmdet3d/version.py" ]] || \
            die "Existing checkout is not MMDetection3D: ${SOURCE_DIR}"
        grep -q "__version__ = '1.4.0'" "${SOURCE_DIR}/mmdet3d/version.py" || \
            die "Existing MMDetection3D checkout is not v1.4.0: ${SOURCE_DIR}"
        printf 'Using existing MMDetection3D v1.4.0 checkout: %s\n' "${SOURCE_DIR}"
    else
        [[ ! -e "${SOURCE_DIR}" ]] || \
            die "Source destination exists and is not a Git checkout: ${SOURCE_DIR}"
        mkdir -p "$(dirname "${SOURCE_DIR}")"
        git clone --branch v1.4.0 --depth 1 --recurse-submodules \
            "${MMDET3D_URL}" "${SOURCE_DIR}"
    fi

    git -C "${SOURCE_DIR}" submodule update --init --recursive

    local fisheye_dir="${SOURCE_DIR}/projects/Fisheye3DOD"
    if [[ -d "${fisheye_dir}/.git" ]]; then
        printf 'Using existing Fisheye3DOD checkout: %s\n' "${fisheye_dir}"
    else
        [[ ! -e "${fisheye_dir}" ]] || \
            die "Fisheye3DOD destination exists but is not a Git checkout: ${fisheye_dir}"
        git clone --recurse-submodules "${FISHEYE_URL}" "${fisheye_dir}"
    fi
    git -C "${fisheye_dir}" submodule update --init --recursive
}

conda_run() {
    "${CONDA_EXE_PATH}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

install_dependencies() {
    local torch_index="https://download.pytorch.org/whl/${CUDA_TAG}"
    export TORCH_CUDA_ARCH_LIST=8.6
    export FORCE_CUDA=1

    conda_run python -m pip install --upgrade 'pip<25' 'setuptools<70' wheel ninja packaging
    conda_run python -m pip install --index-url "${torch_index}" \
        torch==2.1.2 torchvision==0.16.2
    conda_run python -m pip install numpy==1.24.4 'opencv-python<4.12' openmim
    conda_run mim install 'mmengine>=0.7.1,<1.0.0'
    conda_run mim install 'mmcv==2.1.0'
    conda_run mim install 'mmdet>=3.0.0,<3.3.0'

    # Install the universal wheel explicitly. This prevents pip from checking
    # every historical trimesh sdist when resolving requirements/runtime.txt.
    conda_run python -m pip install --only-binary=:all: \
        trimesh==4.12.2

    # Only runtime dependencies are needed for Fisheye training. Do not install
    # requirements/optional.txt here: its Waymo TensorFlow 2.6 package pins
    # NumPy 1.19.x and conflicts with the modern OpenMMLab/trimesh stack.
    conda_run python -m pip install \
        -r "${SOURCE_DIR}/requirements/runtime.txt" trimesh==4.12.2
    conda_run python -m pip install --no-deps -v -e "${SOURCE_DIR}"

    if [[ "${INSTALL_OCAM}" -eq 1 ]]; then
        conda_run python -m pip install \
            'git+https://github.com/matsuren/ocamcalib_undistort.git'
    fi

    conda_run python -m pip install --no-build-isolation --no-deps -v -e \
        "${SOURCE_DIR}/projects/Fisheye3DOD"

    local item
    for item in "${EXTRA_REQUIREMENTS[@]}"; do
        [[ -f "${item}" ]] || die "Requirements file does not exist: ${item}"
        conda_run python -m pip install -r "$(abspath "${item}")"
    done
    for item in "${LOCAL_PACKAGES[@]}"; do
        [[ -d "${item}" ]] || die "Local package does not exist: ${item}"
        conda_run python -m pip install --no-build-isolation -v -e "$(abspath "${item}")"
    done
}

verify_environment() {
    conda_run python -c '
import sys, mmcv, mmdet, mmdet3d, mmengine, torch
assert sys.version_info[:2] == (3, 8), sys.version
assert mmdet3d.__version__ == "1.4.0", mmdet3d.__version__
assert torch.cuda.is_available(), "PyTorch cannot access an NVIDIA GPU"
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} (CUDA runtime {torch.version.cuda})")
print(f"MMCV: {mmcv.__version__}")
print(f"MMEngine: {mmengine.__version__}")
print(f"MMDetection: {mmdet.__version__}")
print(f"MMDetection3D: {mmdet3d.__version__}")
print(f"Visible GPUs: {torch.cuda.device_count()}")'
}

main() {
    install_miniconda
    choose_cuda
    create_environment
    obtain_sources
    install_dependencies
    verify_environment

    printf '\nEnvironment is ready. Activate it with:\n'
    printf '  source %q\n' "$(dirname "$(dirname "${CONDA_EXE_PATH}")")/etc/profile.d/conda.sh"
    printf '  conda activate %q\n' "${ENV_NAME}"
    printf '  cd %q\n' "${SOURCE_DIR}"
}

main
