from collections.abc import Iterator
from time import time
from typing import Any, cast

import numpy
from bitshuffle import (  # pyright: ignore[reportMissingTypeStubs]
    compress_lz4,  # pyright: ignore[reportUnknownVariableType]
)
from cbor import (  # pyright: ignore[reportMissingTypeStubs]
    dumps,  # pyright: ignore[reportUnknownVariableType]
)
from mpi4py import MPI
from numpy.typing import NDArray

from ...models.parameters import (
    SimplonBinarySerializerParameters,
)
from ...utils.logging import log_error_and_exit, log_info
from ...utils.protocols import DataSerializerProtocol
from ...utils.typing import StrFloatIntNDArray


class SimplonBinarySerializer(DataSerializerProtocol):
    """
    See documentation of the `__init__` function.
    """

    # Detector physical constants emitted in the start message (the Jungfrau values used
    # by the LCLS/DIALS collaboration).
    _detector_material: str = "Si"
    _detector_thickness: float = 0.32

    def __init__(self, parameters: SimplonBinarySerializerParameters) -> None:
        """
        Initializes a Simplon data serializer

        This serializers turns a dictionary of numpy arrays into a binary with an
        internal structure of a Simplon message. This serializer follows the 1.8
        version of the Simplon specification (published by Dectris)

        Arguments:

            parameters: The data serializer configuration parameters
        """
        # isinstance (not an exact type-string match) so that subclasses carrying their
        # own discriminator can reuse this initializer via super().__init__.
        if not isinstance(parameters, SimplonBinarySerializerParameters):
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
        # Optional flattened data key of the photon_wavelength PV (e.g.
        # "SIOC:SYS0:ML00:AO192"). When unset, photon_wavelength is reported as 0.
        self._photon_wavelength_source: str | None = parameters.photon_wavelength_source
        # Optional flattened data key of a spectrometer source (e.g.
        # "feespec.raw.hproj"). When unset, no spectrometer fields are added.
        self._spectrometer_source: str | None = parameters.spectrometer_source

    def _compress(self, array: StrFloatIntNDArray) -> bytes:
        """Compress a frame into the wire payload with bitshuffle-lz4."""
        compressed: NDArray[numpy.uint8] = cast(
            NDArray[numpy.uint8], compress_lz4(array, block_size=2**12)
        )
        return compressed.tobytes()

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Whether a value is a back-filled missing reading: a floating array or scalar
        that is entirely non-finite (all-NaN). Valid data short-circuits on the first
        finite element. Non-floating values are never treated as missing here."""
        array: NDArray[Any] = numpy.asarray(value)
        return bool(
            numpy.issubdtype(array.dtype, numpy.floating)
            and not numpy.isfinite(array).any()
        )

    def _photon_wavelength(self, data: dict[str, StrFloatIntNDArray | None]) -> Any:
        """The real photon_wavelength PV value for this event in Angstroms, or 0 when no
        source is configured or the reading is unavailable. The PV (e.g.
        SIOC:SYS0:ML00:AO192) reports nanometers; the consumer works in Angstroms, so
        convert here (1 nm = 10 Angstrom)."""
        if self._photon_wavelength_source is None:
            return 0
        block = data.get(self._photon_wavelength_source)
        if block is None or block[-1] is None or self._is_missing(block[-1]):
            return 0
        return block[-1] * 10.0

    def _beam_fields(
        self, data: dict[str, StrFloatIntNDArray | None]
    ) -> dict[str, Any]:
        """The beam-direction and per-shot photon energy/wavelength fields for an image
        message, or an empty dict when the eBeam fields are absent or back-filled as
        missing (all-NaN) this event, so the consumer falls back rather than using
        fabricated values."""
        try:
            energy: Any = data["ebeamh.raw.ebeamPhotonEnergy"][-1]
            beam: dict[str, Any] = {
                "beam_direction": {
                    "angle_x": data["ebeamh.raw.ebeamUndAngX"][-1],
                    "angle_y": data["ebeamh.raw.ebeamUndAngY"][-1],
                    "position_x": data["ebeamh.raw.ebeamUndPosX"][-1],
                    "position_y": data["ebeamh.raw.ebeamUndPosY"][-1],
                },
                # psana2's undulator-equation conversion of the eBeam L3 energy to the
                # true per-shot photon energy in eV (the consumer divides
                # factor_ev_angstrom by this). Do NOT ship the raw ebeamL3Energy here:
                # that is the electron-beam energy in MeV, not a photon energy.
                "photon_energy": energy,
                "photon_wavelength": self._photon_wavelength(data),
            }
        except KeyError as e:
            log_info(f"Field: {e.args[0]} not found in data_sources. Skipping.")
            return {}
        if self._is_missing(energy):
            return {}
        return beam

    def _spectrometer_fields(
        self, data: dict[str, StrFloatIntNDArray | None]
    ) -> dict[str, Any]:
        """The per-event spectrometer message fields, or an empty dict when no
        spectrometer source is configured or its reading is missing for this event (so
        the frame is still sent). The consumer calibrates pixel -> eV and derives a
        per-shot wavelength from it."""
        if self._spectrometer_source is None:
            return {}
        block = data.get(self._spectrometer_source)
        if block is None:
            return {}
        array = block[-1]
        # An absent or back-filled (all-NaN) reading is treated as missing so the
        # consumer falls back rather than calibrating garbage.
        if array is None or self._is_missing(array):
            return {}
        return {
            "spectrometer_compressed_data": self._compress(array),
            "spectrometer_dtype": str(array.dtype),
            "spectrometer_shape": "x".join(map(str, array.shape)),
        }

    def _build_start_message(
        self, data: dict[str, StrFloatIntNDArray | None], array: StrFloatIntNDArray
    ) -> dict[str, Any]:
        """Build the one-per-run Simplon start message."""
        return {
            "type": "start",
            # LCLS uses "run" terminology where the Simplon API uses "series_id"; we
            # deliberately diverge from the API and ship "run_id".
            "run_id": data["run_number"][-1],
            "start_time": data["run_timestamp"][-1],
            "duration": "N/A",
            "beamline": data["source_identifier"][-1][3][4:7].upper(),
            "experiment": data["experiment"][-1],
            "beam_type": "X-ray",
            "polarization": {
                "fraction": self._polarization.get("polarization_fraction", 0),
                "axis": self._polarization.get(
                    "polarization_axis", [0.0, 0.0, 0.0]
                ),
            },
            "data_collection_rate": self._data_rate,
            "image_dtype": str(array.dtype),
            "shape": "x".join(map(str, array.shape)),
            "algorithm": "bitshuffle-lz4",
            "detector": {
                "name": self._detector_name,
                "id": data["jungfrau._detid"][-1],
                "type": self._detector_type,
                "geometry": data["jungfrau.raw._det_geotxt_default"][-1],
                "pixel_coords": numpy.array(
                    data["jungfrau.raw._pixel_coords"][-1]
                ).tobytes()
                if "jungfrau.raw._pixel_coords" in data
                else "",
                "material": self._detector_material,
                "thickness": self._detector_thickness,
            },
            "message_id": self._node_rank * 10000 + self._rank_message_count,
            "timestamp": time(),
        }

    def _build_image_message(
        self, data: dict[str, StrFloatIntNDArray | None], array: StrFloatIntNDArray
    ) -> dict[str, Any]:
        """Build a per-event Simplon image message."""
        return {
            "type": "image",
            "run_id": data["run_number"][-1],
            "compressed_data": self._compress(array),
            **self._beam_fields(data),
            **self._spectrometer_fields(data),
            "image_dtype": str(array.dtype),
            # .item() -> native Python scalar: cbor serializes Python int/float and
            # numpy float64 (a float subclass), but NOT a numpy float32 scalar, which
            # is what array.sum() yields for float32 calibrated (keV) data.
            "sum": array.sum().item(),
            "message_id": self._node_rank * 10000 + self._rank_message_count,
            "timestamp": cast(NDArray[numpy.str_], data["timestamp"])[-1],
        }

    def _build_stop_message(self, run_id: Any) -> dict[str, Any]:
        """Build the one-per-run Simplon end message."""
        return {
            "type": "end",
            "run_id": run_id,
            "timestamp": time(),
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
        # Only the last rank emits the per-run start/end markers.
        must_send_first_message: bool = self._node_rank == self._node_pool_size - 1
        run_id: Any = 0
        # Per-rank accounting so loss is never silent.
        received: int = 0
        images: int = 0
        dropped: int = 0

        data: dict[str, StrFloatIntNDArray | None]
        for data in stream:
            received += 1
            try:
                data_block = data[self._data_source_to_serialize]
            except KeyError:
                log_error_and_exit(
                    f"The {self._data_source_to_serialize} data source, that the "
                    "SimplonBinarySerializer is supposed to serialize, cannot be found in"
                    "the data"
                )

            if data_block is None:
                # The primary frame is absent for this event; skip it. The frame must be
                # present to serialize an image, so an event lacking it is dropped.
                dropped += 1
                continue

            array: StrFloatIntNDArray = data_block[-1]

            # Accept any int/float width: Jungfrau raw.calib is float32. The exact dtype
            # is carried in the start/image messages and honored by the consumer.
            if not (
                numpy.issubdtype(array.dtype, numpy.integer)
                or numpy.issubdtype(array.dtype, numpy.floating)
            ):
                log_error_and_exit(
                    f"The {self._data_source_to_serialize} data source is not of type int "
                    "or float, as required by the SimplonBinarySerializer"
                )

            run_id = data["run_number"][-1]

            if must_send_first_message:
                yield cast(bytes, dumps(self._build_start_message(data, array)))
                must_send_first_message = False

            yield cast(bytes, dumps(self._build_image_message(data, array)))
            self._rank_message_count += 1
            images += 1

        if self._node_rank == self._node_pool_size - 1:
            yield cast(bytes, dumps(self._build_stop_message(run_id)))

        log_info(
            f"[Rank {self._node_rank}] serializer done: received={received} "
            f"images={images} dropped={dropped} (images + dropped == received: "
            f"{images + dropped == received})"
        )
