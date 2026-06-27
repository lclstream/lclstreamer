from pathlib import Path
from typing import Dict, List, Literal, Self, Union, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from typing_extensions import Annotated


class _CustomBaseModel(BaseModel):
    # A base Pydantic model that forbids extra fields by default

    model_config = ConfigDict(
        extra="forbid",
    )


####### Event Sources ########


class InternalEventSourceParameters(_CustomBaseModel):
    """
    Configuration parameters for the Internal Event Source

    This event source generates synthetic events entirely in-process and does
    not depend on any external data-acquisition framework. It is intended
    mainly for testing and development

    Attributes:

        type: Discriminator field, must be ``"InternalEventSource"``

        number_of_events_to_generate: Total number of synthetic events to
            produce before the source is exhausted
    """

    type: Literal["InternalEventSource"]
    number_of_events_to_generate: int
    model_config = ConfigDict(extra="allow")


class Psana1EventSourceParameters(_CustomBaseModel):
    """
    Configuration parameters for the Psana1 Event Source

    This event source reads events using the psana1 framework (LCLS-I)

    Attributes:

        type: Discriminator field, must be ``"Psana1EventSource"``
    """

    type: Literal["Psana1EventSource"]


class Psana2EventSourceParameters(_CustomBaseModel):
    """
    Configuration parameters for the Psana2 Event Source

    This event source reads events using the psana2 framework (LCLS-II)

    Attributes:

        type: Discriminator field, must be ``"Psana2EventSource"``
    """

    type: Literal["Psana2EventSource"]


EventSourceParameters = Annotated[
    Union[
        InternalEventSourceParameters,
        Psana1EventSourceParameters,
        Psana2EventSourceParameters,
    ],
    Field(discriminator="type"),
]


###### Data Sources #######

class GenericRandomNumpyArrayParameters(_CustomBaseModel):
    """
    Parameters for the GenericRandomNumpyArray class

    """
    type: Literal["GenericRandomNumpyArray"]
    array_shape: int | tuple[int, ...]
    array_dtype: str
    always_random: bool = True

    @field_validator("array_shape", mode="before")
    @classmethod
    def convert_int_to_tuple(cls, v):
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return tuple(int(p) for p in parts)
        if isinstance(v, int):
            return (v,)
        return v

class ConstValueParameters(_CustomBaseModel):
    """
    Parameters for ConstValue class
    """
    type: Literal["ConstValue"]
    value: int | float | list[int | float]
    dtype: str

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: Any):
        if isinstance(v, str): # "6," -> [6]
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if len(parts) == 1:
                return int(parts[0]) if parts[0].isdigit() else float(parts[0])
            return [
                int(p) if p.isdigit() else float(p)
                for p in parts
            ]
        return v

class _PsanaDetectorInterfaceParameters(_CustomBaseModel):
    psana_name: str
    psana_fields: list[str] | str | None = None

    @model_validator(mode="after")
    def validate_fields(self):
        if ":" not in self.psana_name and self.psana_fields is None:
            raise ValueError(
                "psana_fields must be specified when psana_name is not a PV."
            )
        return self

class Psana1DetectorInterfaceParameters(_PsanaDetectorInterfaceParameters):
    type: Literal["Psana1DetectorInterface"]

class Psana2DetectorInterfaceParameters(_PsanaDetectorInterfaceParameters):
    type: Literal["Psana2DetectorInterface"]
    dtype: str | None = None

class Psana2TimestampParameters(_CustomBaseModel):
    """
    Parameters for psana2 timestamp interface
    """
    type: Literal["Psana2Timestamp"]

class Psana1TimestampParameters(_CustomBaseModel):
    """
    Parameters for psana1 timestamp interface
    """
    type: Literal["Psana1Timestamp"]

class SourceIdentifierParameters(_CustomBaseModel):
    """
    Parameters for source identifier data source interface
    """
    type: Literal["SourceIdentifier"]

class Psana2RunInfoParameters(_CustomBaseModel):
    """
    Parameters for run info data source interface
    """
    type: Literal["Psana2RunInfo"]

DataSourceParameters = Annotated[
    Union[
        GenericRandomNumpyArrayParameters,
        ConstValueParameters,
        Psana1DetectorInterfaceParameters,
        Psana2DetectorInterfaceParameters,
        Psana1TimestampParameters,
        Psana2TimestampParameters,
        SourceIdentifierParameters,
        Psana2RunInfoParameters,
    ],
    Field(discriminator="type"),
]


####### Processing Pipelines #########


class BatchProcessingPipelineParameters(_CustomBaseModel):
    """
    Configuration parameters for the Batch Processing Pipeline

    This pipeline accumulates individual events into fixed-size batches before
    passing them downstream

    Attributes:

        type: Discriminator field, must be ``"BatchProcessingPipeline"``

        batch_size: Number of events to accumulate per batch
    """

    type: Literal["BatchProcessingPipeline"]
    batch_size: int


class PeaknetPreprocessingPipelineParameters(_CustomBaseModel):
    """
    Configuration parameters for the PeakNet Preprocessing Pipeline

    This pipeline pads detector images to a uniform size, accumulates them
    into batches, and optionally adds a channel dimension, preparing the data
    for inference with the PeakNet model

    Attributes:

        type: Discriminator field, must be ``"PeaknetPreprocessingPipeline"``

        batch_size: Number of events to accumulate per batch

        target_height: Target image height (in pixels) after padding

        target_width: Target image width (in pixels) after padding

        pad_style: How to distribute padding around the image; either
            ``"center"`` (equal padding on both sides) or ``"bottom-right"``
            (padding added only to the bottom and right edges)
            Defaults to ``"center"``

        add_channel_dim: Whether to insert a channel dimension after batching,
            converting (B, H, W) arrays to (B, C, H, W). Defaults to ``True``

        num_channels: Number of channels to produce when ``add_channel_dim``
            is ``True``. Defaults to ``1``
    """

    type: Literal["PeaknetPreprocessingPipeline"]
    batch_size: int
    target_height: int
    target_width: int
    pad_style: Literal["center", "bottom-right"] = "center"
    add_channel_dim: bool = True
    num_channels: int = 1


ProcessingPipelineParameters = Annotated[
    Union[BatchProcessingPipelineParameters, PeaknetPreprocessingPipelineParameters],
    Field(discriminator="type"),
]


####### Serializers ##########


class SimplonBinarySerializerParameters(_CustomBaseModel):
    """
    Configuration parameters for the Simplon binary serializer

    This serializer encodes event data into the Simplon 1.8 binary message
    format as specified by Dectris

    Attributes:

        type: Discriminator field, must be ``"SimplonBinarySerializer"``

        data_source_to_serialize: Name of the data source whose array will be
            compressed and embedded in each Simplon image message

        polarization_fraction: Fraction of linear polarization of the X-ray
            beam (between 0 and 1)

        polarization_axis: Three-element list representing the polarization
            axis direction vector

        data_collection_rate: Human-readable string describing the nominal
            data collection rate (e.g. ``"120 Hz"``)

        detector_name: Human-readable name of the detector

        detector_type: Model or type string identifying the detector hardware
    """

    type: Literal["SimplonBinarySerializer"]
    data_source_to_serialize: str
    polarization_fraction: float
    polarization_axis: List[float]
    data_collection_rate: str
    detector_name: str
    detector_type: str


class HDF5BinarySerializerParameters(_CustomBaseModel):
    """
    Configuration parameters for the HDF5 binary serializer

    This serializer encodes a batch of event data arrays into an in-memory
    HDF5 file. Optional compression can be applied to each dataset

    Attributes:

        type: Discriminator field, must be ``"HDF5BinarySerializer"``

        compression_level: Compression level passed to the chosen algorithm.
            Interpretation depends on the algorithm. Defaults to ``3``

        compression: Compression algorithm to use. Supported values are
            ``"gzip"``, ``"gzip_with_shuffle"``, ``"bitshuffle_with_lz4"``,
            ``"bitshuffle_with_zstd"``, and ``"zfp"``. Set to ``None`` to
            disable compression. Defaults to ``None``

        fields: Dictionary storing the mapping from data source name to the
            HDF5 dataset path under which that source's data will be stored
    """

    type: Literal["HDF5BinarySerializer"]
    compression_level: int = 0
    compression: (
        Literal[
            "gzip",
            "gzip_with_shuffle",
            "bitshuffle_with_lz4",
            "bitshuffle_with_zstd",
            "zfp",
        ]
        | None
    ) = None
    fields: dict[str, str]


DataSerializerParameters = Annotated[
    Union[HDF5BinarySerializerParameters, SimplonBinarySerializerParameters],
    Field(discriminator="type"),
]


######### Data Handlers #################


class BinaryDataStreamingDataHandlerParameters(_CustomBaseModel):
    """
    Configuration parameters for the Binary Data Streaming Data Handler

    This data handler forwards serialized byte objects to one or more remote
    endpoints over a ZMQ PUSH socket

    Attributes:

        type: Discriminator field, must be ``"BinaryDataStreamingDataHandler"``

        urls: List of endpoint URLs to bind to (server mode) or connect to
            (client mode)

        distribute: Boolean, if True: round robin connect the ranks, False: all ranks connect to all urls and send the same data

        buffer: buffer size, if set to 0 the OS default is used

        role: Whether this node acts as the ZMQ ``"server"`` (binds) or
            ``"client"`` (connects). Defaults to ``"server"``

        library: Underlying transport library to use. Currently only ``"zmq"``
            is supported. Defaults to ``"zmq"``

        socket_type: Socket pattern to use. Currently only ``"push"`` is
            supported. Defaults to ``"push"``
    """

    type: Literal["BinaryDataStreamingDataHandler"]
    urls: List[str]
    distribute: bool
    buffer: int
    role: Literal["server", "client"] = "client"
    library: Literal["zmq"] = "zmq"
    socket_type: Literal["push"] = "push"


class BinaryFileWritingDataHandlerParameters(_CustomBaseModel):
    """
    Configuration parameters for the Binary File Writing Data Handler

    This data handler writes each serialized byte object to a separate file on
    the filesystem

    Attributes:

        type: Discriminator field, must be ``"BinaryFileWritingDataHandler"``

        file_prefix: Optional string prepended to every output filename,
            separated from the rest of the name by an underscore. Defaults to
            ``""`` (no prefix)

        file_suffix: File extension used for output files, without the leading
            dot. Defaults to ``"h5"``

        write_directory: Directory in which output files are created. The
            directory is created (including parents) if it does not already
            exist. Defaults to the current working directory
    """

    type: Literal["BinaryFileWritingDataHandler"]
    file_prefix: str = ""
    file_suffix: str = "h5"
    write_directory: Path = Path.cwd()


DataHandlerParameters = Annotated[
    Union[
        BinaryDataStreamingDataHandlerParameters, BinaryFileWritingDataHandlerParameters
    ],
    Field(discriminator="type"),
]


class Parameters(_CustomBaseModel):
    """
    Top-level configuration parameters for an lclstreamer run

    This model aggregates all sub-component configuration into a single object

    Attributes:

        source_identifier: A string that uniquely identifies the data source
            (e.g. a psana data source string)

        skip_incomplete_events: When ``True``, events for which one or more
            data sources returned no data are silently dropped from the stream

        event_source: Configuration for the event source

        data_sources: Mapping from arbitrary data source names to the
            configuration parameters of each data source

        processing_pipeline: Configuration for the Processing Pipeline applied
            to the event stream

        data_serializer: Configuration for the serializer that converts
            processed events into byte objects

        data_handlers: Ordered list of data handler configurations; each
            handler receives the serialized byte object in turn
    """

    source_identifier: str
    skip_incomplete_events: bool

    event_source: EventSourceParameters
    data_sources: Dict[str, DataSourceParameters]
    processing_pipeline: ProcessingPipelineParameters
    data_serializer: DataSerializerParameters
    data_handlers: List[DataHandlerParameters]

    @model_validator(mode="after")
    def _check_model(self) -> Self:
        # Validates cross-field constraints after model initialization

        if self.data_serializer.type == "SimplonBinarySerializer":
            required_sources = [
                "timestamp",
                "detector_data",
                # "photon_wavelength",
                "detector_geometry",
                "run_info",
            ]
            source_missing = [
                k for k in required_sources if k not in self.data_sources.keys()
            ]
            if source_missing:
                raise ValueError(
                    f"Required fields: {source_missing} is missing from data_sources "
                    "for SimplonBinarySerializer."
                )

        return self
