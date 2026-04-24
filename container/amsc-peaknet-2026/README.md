# Container for the AmSC Demo 2026

Build it *from the top level directory of this repo* with:

``` bash
podman build -t lclstreamer-psana1:23990c0 --build-arg PSANA_VERSION=1nersc --build-arg COMMIT=d5c70c5 -f container/amsc-peaknet-2026/Containerfile .
```
