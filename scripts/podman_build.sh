#!/bin/bash

GIT_SHORT_HASH=$(git rev-parse --short HEAD)
ROOT_DIR=$(git rev-parse --show-toplevel)
CONTAINERFILE_PATH="${ROOT_DIR}/container/amsc-peaknet-2026/Containerfile"

podman-hpc build \
    -t registry.nersc.gov/lcls/lclstreamer:nersc-peaknet-${GIT_SHORT_HASH} \
    -t registry.nersc.gov/lcls/lclstreamer:nersc-peaknet-latest \
    -f "${CONTAINERFILE_PATH}" "${ROOT_DIR}"