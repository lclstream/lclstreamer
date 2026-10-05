from collections.abc import Iterator
from time import time
from typing import Any, cast

import numpy
from cbor import (  # pyright: ignore[reportMissingTypeStubs]
    dumps,  # pyright: ignore[reportUnknownVariableType]
)
from mpi4py import MPI
from numpy.typing import NDArray

from ...models.parameters import (
    SimplonBinarySerializerParameters,
)
from ...processing_pipelines.common.data_storage import is_null_value
from ...utils.logging import log_error_and_exit, log_info
from ...utils.protocols import DataSerializerProtocol
from ...utils.typing import StrFloatIntNDArray


class SimplonBinarySerializer(DataSerializerProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(self, parameters: SimplonBinarySerializerParameters) -> None:
        """
        Initializes a Simplon data serializer

        This serializers turns a dictionary of numpy arrays into a binary with an
        internal structure of a Simplon message. This serializer follows the 1.8
        version of the Simplon specification (published by Dectris)

        Arguments:

            parameters: The data serializer configuration parameters
        """
        if parameters.type != "SimplonBinarySerializer":
            log_error_and_exit(
                "Data serializer parameters do not match the expected type"
            )
        self._data_source_to_serialize: str = parameters.data_source_to_serialize
        self._polarization: dict[str, Any] = {
            "polarization_fraction": parameters.polarization_fraction,
            "polarization_axis": parameters.polarization_axis,
        }
        self._data_rate: str = parameters.data_collection_rate
        self._detector_name: str = parameters.detector_name
        self._detector_type: str = parameters.detector_type
        self._node_rank: int = MPI.COMM_WORLD.Get_rank()
        self._node_pool_size: int = MPI.COMM_WORLD.Get_size()
        self._rank_message_count: int = 1
        self._photon_wavelength_source: str | None = parameters.photon_wavelength_source
        self._spectrometer_source: str | None = parameters.spectrometer_source

    def _photon_wavelength(self, data: dict[str, StrFloatIntNDArray | None]) -> Any:
        """The photon_wavelength PV in Angstrom (the PV reports nm), or 0 if missing"""
        if self._photon_wavelength_source is None:
            return 0
        block: StrFloatIntNDArray | None = data.get(self._photon_wavelength_source)
        if block is None or is_null_value(block[-1]):
            return 0
        return block[-1] * 10.0

    def _spectrometer_fields(
        self, data: dict[str, StrFloatIntNDArray | None]
    ) -> dict[str, Any]:
        """The spectrometer fields of an image message, or none if missing"""
        if self._spectrometer_source is None:
            return {}
        block: StrFloatIntNDArray | None = data.get(self._spectrometer_source)
        if block is None or is_null_value(block[-1]):
            return {}
        spectrum: StrFloatIntNDArray = block[-1]
        return {
            "spectrometer_data": spectrum.tobytes(),
            "spectrometer_dtype": str(spectrum.dtype),
            "spectrometer_shape": "x".join(map(str, spectrum.shape)),
        }

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray | None]]
    ) -> Iterator[bytes]:
        """
        Serializes data to a binary blob with an internal Simplon message structure

        Arguments:

            source: A dictionary storing event data

        Yields:

            byte_block: A bytes object
        """

        must_send_first_message: bool = False
        if self._node_rank == self._node_pool_size - 1:
            must_send_first_message = True
        run_number: int = 0

        data: dict[str, StrFloatIntNDArray | None]
        for data in stream:
            # Absent from the batch when the frame has been missing from every event so far
            if (data_block := data.get(self._data_source_to_serialize)) is None:
                log_info(f"Skipping event missing {self._data_source_to_serialize}")
                continue
            array: StrFloatIntNDArray = data_block[-1]

            if is_null_value(array):
                log_info(f"Skipping event missing {self._data_source_to_serialize}")
                continue

            if not (
                numpy.issubdtype(array.dtype, numpy.integer)
                or numpy.issubdtype(array.dtype, numpy.floating)
            ):
                log_error_and_exit(
                    f"The {self._data_source_to_serialize} data source is not of type int "
                    "or float, as required by the SimplonBinarySerializer"
                )

            run_number: str = data["run_number"][-1]

            if self._node_rank == self._node_pool_size - 1:
                if must_send_first_message:

                    yield cast(
                        bytes,
                        dumps(
                            {
                                "type": "start",
                                "run_id": run_number,
                                "start_time": data["run_timestamp"][-1],
                                "duration": "N/A",
                                "beamline": data["source_identifier"][-1][3][4:7].upper(),
                                "experiment": data["experiment"][-1],
                                "beam_type": "X-ray",
                                "polarization": {
                                    "fraction": self._polarization.get(
                                        "polarization_fraction", 0
                                    ),
                                    "axis": self._polarization.get(
                                        "polarization_axis", [0.0, 0.0, 0.0]
                                    ),
                                },
                                "data_collection_rate": self._data_rate,
                                "image_dtype": str(array.dtype),
                                "shape": "x".join(map(str, array.shape)),
                                "detector": {
                                    "name": self._detector_name,
                                    "id": data["jungfrau._detid"][-1],
                                    "type": self._detector_type,
                                    "geometry": data["jungfrau.raw._det_geotxt_default"][-1],
                                    "pixel_coords": numpy.array(
                                        data["jungfrau.raw._pixel_coords"][-1]
                                    ).tobytes()
                                    if "jungfrau.raw._pixel_coords" in data else "",
                                    "material": "???",
                                    "thickness": "???",
                                },
                                "message_id": self._node_rank * 10000
                                + self._rank_message_count,
                                "timestamp": time(),
                            }
                        ),
                    )
                    self._rank_message_count += 1

            must_send_first_message = False

            # .item(): cbor cannot encode the numpy float32 scalar of a float32 frame
            array_sum: float | int = array.sum().item()

            beam_data_dict: dict[str, Any] = {}
            try:
                beam_data_dict = {
                    "beam_direction": {
                        "angle_x": data["ebeamh.raw.ebeamUndAngX"][-1],
                        "angle_y": data["ebeamh.raw.ebeamUndAngY"][-1],
                        "position_x": data["ebeamh.raw.ebeamUndPosX"][-1],
                        "position_y": data["ebeamh.raw.ebeamUndPosY"][-1],
                    },
                    "photon_energy": data["ebeamh.raw.ebeamPhotonEnergy"][-1],
                    "photon_wavelength": 0,  # data["photon_wavelength"][-1]
                }
            except KeyError as e:
                log_info(f"Field: {e.args[0]} not found in data_sources. Skipping.")
            if self._photon_wavelength_source is not None:
                beam_data_dict["photon_wavelength"] = self._photon_wavelength(data)

            message: dict[str, Any] = {
                "type": "image",
                "run_id": run_number,
                "data": array.tobytes(),
                **beam_data_dict,
                **self._spectrometer_fields(data),
                "image_dtype": str(array.dtype),
                "sum": array_sum,
                "message_id": self._node_rank * 10000 + self._rank_message_count,
                "timestamp": cast(NDArray[numpy.str_], data["timestamp"])[-1],
            }
            yield cast(bytes, dumps(message))
            self._rank_message_count += 1

        if self._node_rank == self._node_pool_size - 1:
            yield cast(
                bytes,
                dumps(
                    {
                        "type": "end",
                        "run_id": run_number,
                        "timestamp": time(),
                    }
                ),
            )
