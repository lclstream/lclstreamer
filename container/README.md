# Container for lclstreamer

This builds an OCI image of `lclstreamer` for a chosen pixi environment.
Select the environment with the `PSANA_VERSION` build arg, which maps to the
`psana<PSANA_VERSION>` pixi environment defined in `pyproject.toml`.

```bash
podman build \
  -t lclstreamer-psana2extmpi:latest \
  --build-arg PSANA_VERSION=2extmpi \
  -f container/Containerfile .
```

## CI

Builds the `psana2extmpi` environment and pushes it to GitHub Container Registry as `ghcr.io/lclstream/lclstreamer-psana2extmpi`

## Running on S3DF

On S3DF, when using apptainer, run with:

```bash
apptainer run --env TMPDIR=/tmp --env OMPI_MCA_orte_tmpdir_base=/tmp <etc>
```

or:

```bash
OMPI_MCA_orte_tmpdir_base=/tmp mpirun -np 5 apptainer run <etc>
```
