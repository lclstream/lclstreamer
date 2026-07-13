from importlib.util import find_spec
from pathlib import Path
from typing import Any, Generator, cast

import numpy
import pytest
from numpy.typing import NDArray

from lclstreamer.models.parameters import DataSourceParameters, Psana2DetectorInterfaceParameters

test_path = Path("/sdf/data/lcls/ds/mfx/mfx100852324/xtc/smalldata/")
try:
    test_path.stat()
    can_access = True
except (PermissionError, FileNotFoundError):
    can_access = False

psana_found: bool = find_spec("psana") is not None


using_psana2: bool = False
if psana_found:
    try:
        from psana import (
            DataSource,  # pyright: ignore[reportUnusedImport, reportAttributeAccessIssue, reportUnknownVariableType]
            xtc_version,  # pyright: ignore[reportUnusedImport, reportAttributeAccessIssue, reportUnknownVariableType]  # noqa: F401
        )

        using_psana2 = True
    except ImportError:
        pass


@pytest.mark.skipif(not psana_found, reason="Test requires psana2")
@pytest.mark.skipif(not using_psana2, reason="Test requires psana2")
@pytest.mark.skipif(not can_access, reason="Inaccessible data")
def test_detector_interface() -> None:

    from lclstreamer.event_data_sources.psana2.data_sources import (
        Psana2DetectorInterface,
    )

    # Psana2AreaDetector
    exp: str = "mfx100852324"
    run: int = 355

    source_identifier: str = f"exp={exp},run={run}"
    ds: Any = (  # pyright: ignore[reportUnknownVariableType]
        DataSource(exp=exp, run=run)  # pyright: ignore[reportPossiblyUnboundVariable]
    )
    psana_run: Any = (  # pyright: ignore[reportUnknownVariableType]
        next(ds.runs())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    event_source: Generator[Any, None, None] = cast(
        Generator[Any, None, None],
        psana_run.events(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    event: Any = next(event_source)
    name: str = "Psana2AreaDetector"
    parameters: Psana2DetectorInterfaceParameters = Psana2DetectorInterfaceParameters(
                                                        type="Psana2DetectorInterface",
                                                        psana_name="jungfrau",
                                                        psana_fields="raw.raw",
                                                    )
    Psana2AreaDetector: Psana2DetectorInterface = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2AreaDetector_func: NDArray[numpy.signedinteger[Any]] = (
        Psana2AreaDetector.get_data(event)["jungfrau.raw.raw"]
    )
    Psana2AreaDetector_raw: NDArray[numpy.signedinteger[Any]] = cast(
        NDArray[numpy.signedinteger[Any]],
        psana_run.Detector("jungfrau").raw.raw(event),  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(Psana2AreaDetector_func, Psana2AreaDetector_raw)

    # Psana 2 Calib
    name = "Psana2Calib"
    parameters = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="jungfrau",
                                                   psana_fields="raw.calib",
                                                  )
    Psana2Calib = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2Calib_func: NDArray[numpy.floating[Any]] = (
        Psana2Calib.get_data(event)["jungfrau.raw.calib"]
    )
    Psana2Calib_raw: NDArray[numpy.floating[Any]] = (  # pyright: ignore[reportUnknownVariableType]
        psana_run.Detector("jungfrau").raw.calib(event)  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(
        Psana2Calib_func,
        Psana2Calib_raw,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Psana2AssembledAreaDetector
    name = "Psana2AssembledAreaDetector"
    parameters = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="jungfrau",
                                                   psana_fields="raw.image",
                                                  )
    Psana2AssembledAreaDetector = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2AssembledAreaDetector_func: NDArray[numpy.floating[Any]] = (
        Psana2AssembledAreaDetector.get_data(event)["jungfrau.raw.image"]
    )
    Psana2AssembledAreaDetector_raw: NDArray[numpy.floating[Any]] = (  # pyright: ignore[reportUnknownVariableType]
        psana_run.Detector("jungfrau").raw.image(event)  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(
        Psana2AssembledAreaDetector_func,
        Psana2AssembledAreaDetector_raw,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Psana2PV
    name = "Psana2PV"
    parameters = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="SIOC:SYS0:ML00:AO192",
                                                  )
    Psana2PV = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2PV_func: NDArray[numpy.floating[Any]] = Psana2PV.get_data(event)["SIOC:SYS0:ML00:AO192"]
    Psana2PV_raw: NDArray[numpy.floating[Any]] = cast(  # pyright: ignore[reportUnknownVariableType]
        NDArray[numpy.floating[Any]],
        psana_run.Detector("SIOC:SYS0:ML00:AO192")(event),  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(
        Psana2PV_func,
        Psana2PV_raw,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Psana2EBeam
    name = "Psana2EBeam"
    parameters = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="ebeamh",
                                                   psana_fields="raw.ebeamL3Energy",
                                                  )
    Psana2EBeam = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2EBeam_func: NDArray[numpy.floating[Any]] = Psana2EBeam.get_data(event)["ebeamh.raw.ebeamL3Energy"]
    Psana2EBeam_raw: NDArray[numpy.floating[Any]] = (  # pyright: ignore[reportUnknownVariableType]
        psana_run.Detector("ebeamh").raw.ebeamL3Energy(event)  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(
        Psana2EBeam_func,
        Psana2EBeam_raw,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Psana2Gmd
    exp = "tmox1016823"
    run = 44

    source_identifier = f"exp={exp},run={run}"

    ds = (  # pyright: ignore[reportUnknownVariableType]
        DataSource(exp=exp, run=run)  # pyright: ignore[reportPossiblyUnboundVariable]
    )
    psana_run = (  # pyright: ignore[reportUnknownVariableType]
        next(ds.runs())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )

    event_source = cast(
        Generator[Any, None, None],
        psana_run.events(),  # pyright: ignore[reportUnknownMemberType]
    )
    event = next(event_source)
    name = "Psana2Gmd"
    parameters = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="gmd",
                                                   psana_fields="raw.milliJoulesPerPulse",
                                                  )
    Psana2Gmd = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2Gmd_func: NDArray[numpy.floating[Any]] = Psana2Gmd.get_data(event)["gmd.raw.milliJoulesPerPulse"]
    Psana2Gmd_raw: NDArray[numpy.floating[Any]] = (  # pyright: ignore[reportUnknownVariableType]
        psana_run.Detector("gmd").raw.milliJoulesPerPulse(event)  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(
        Psana2Gmd_func,
        Psana2Gmd_raw,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Psana2Camera
    name = "Psana2Camera"
    parameters = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="tmo_fzppiranha",
                                                   psana_fields="raw.raw",
                                                  )
    Psana2Camera = Psana2DetectorInterface(
        name=name,
        parameters=parameters,
        additional_info={
            "run": psana_run,
            "source_identifier": source_identifier,
        },
    )
    Psana2Camera_func: NDArray[numpy.floating[Any]] = Psana2Camera.get_data(event)["tmo_fzppiranha.raw.raw"]
    Psana2Camera_raw: NDArray[numpy.floating[Any]] = (  # pyright: ignore[reportUnknownVariableType]
        psana_run.Detector("tmo_fzppiranha").raw.raw(event)  # pyright: ignore[reportUnknownMemberType]
    )
    assert numpy.array_equal(
        Psana2Camera_func,
        Psana2Camera_raw,  # pyright: ignore[reportUnknownArgumentType]
    )
