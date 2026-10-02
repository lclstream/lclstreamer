from collections.abc import Iterator

from dataclasses import dataclass, asdict

import numpy

from ...models.parameters import (
    PeaknetPreprocessingPipelineParameters,
)
from ...utils.logging import log_error_and_exit
from ...utils.protocols import ProcessingPipelineProtocol
from ...utils.typing import StrFloatIntNDArray


@dataclass
class _PeakList:
    # This typed dictionary stores information about a set of peaks found by a
    # peak-finding algorithm in a detector data frame.
    num_peaks: int
    fs: list[float]
    ss: list[float]
    intensity: list[float]
    num_pixels: list[float]
    max_pixel_intensity: list[float]
    snr: list[float]


class CrystfelPreprocessingPipeline(ProcessingPipelineProtocol):
    """
    See documentation of the `__init__` function
    """

    def __init__(self, parameters: PeaknetPreprocessingPipelineParameters) -> None:
        """
        Initializes a CrystFEL Preprocessing Pipeline

        This pipeline performs all the preprocessing necessary for the data to be
        cosumed by the CrystFEL crystallography application:

        1. An empy peak list is added to the data
        2. Detector data is "slabified" (squashed along the first dimension)
        3. Add a channel dimension to the batched image data (after batching)

        Arguments:

            parameters: The processing pipeline configuration parameters
        """
        if parameters.type != "CrystfelPreprocessingPipeline":
            log_error_and_exit(
                "Processing pipeline parameters do not match the expected type"
            )
        self._batch_size: int = 1

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray | None]]
    ) -> Iterator[dict[str, StrFloatIntNDArray | None]]:
        """
        Applies the PeakNet Preprocessing Pipeline to incoming event data

        For each event, the steps of the processing pipeline are applied to the
        incoming data. Any remaining events that do not fill a complete batch at
        the end of the data stream are yielded as a partial batch.

        Arguments:

            stream: A dictionary storing event data

        Yields:

            batch: A dictionary of processed and batched events
        """
        empty_peak_list: dict[str, int | list[float]] = asdict(
            _PeakList(
                num_peaks=0,
                fs=[],
                ss=[],
                intensity=[],
                num_pixels=[],
                max_pixel_intensity=[],
                snr=[],
            )
        )

        preprocessed_data: dict[str, StrFloatIntNDArray | None] = {}

        h: float = 6.626070e-34  # J.m
        c: float = 2.99792458e8  # m/s
        joules_per_ev: float = 1.602176621e-19  # J/eV

        data: dict[str, StrFloatIntNDArray | None]
        for data in stream:
            detector_data_shape: tuple[int, ...] = data["detector_data"]["detector_data"].shape
            preprocessed_data["detector_data"] = data["detector_data"]["detector_data"].reshape(
                detector_data_shape[0] * detector_data_shape[1],
                *detector_data_shape[2:],
            )
            preprocessed_data["peak_list"] = empty_peak_list
            preprocessed_data["beam_energy"]: float = (h / joules_per_ev * c) / (
                data["photon_wavelength"]["photon_wavelength"].item() * 1e-9
            )
            preprocessed_data["detector_distance"] = data["detector_distance"]["detector_distance"].item()
            preprocessed_data["event_id"] = data["timestamp"].astype(numpy.int64).item()
            preprocessed_data["timestamp"] = (
                data["timestamp"].astype(numpy.int64).item()
            )
            if "optical_laser_active" in data:
                preprocessed_data["optical_laser_active"] = data[
                    "optical_laser_active"
                ].item()
            else:
                preprocessed_data["optical_laser_active"] = False
            preprocessed_data["source"] = data["run_info"]["source_identifier"].item()

            yield preprocessed_data
