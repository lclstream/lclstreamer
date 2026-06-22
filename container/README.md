# Container for the AmSC Demo 2026

Build it *from the top level directory of this repo* with:

``` bash
podman build -t lclstreamer-psana2:<COMMIT> --build-arg PSANA_VERSION=1nersc [--build-arg COMMIT=<COMMIT>] -f container/Containerfile .
```

On S3DF, when using apptainer, run with:

apptainer run --env TMPIDR=/tmp --env OMPI_MCA_orte_tmpdir_base=/tmp <etc>

or:

OMPI_MCA_orte_tmpdir_base=/tmp  mpirun -np 5 apptainer run <etc>
