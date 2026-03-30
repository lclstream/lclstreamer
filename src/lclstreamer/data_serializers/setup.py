from ..models.parameters import (
    DataSerializerParameters,
    Parameters,
)
from ..utils.logging import log_error_and_exit
from ..utils.protocols import DataSerializerProtocol
from .dectris.simplon import SimplonBinarySerializer as SimplonBinarySerializer
from .files.fast_binary import FastBinarySerializer as FastBinarySerializer
from .files.hdf5 import HDF5BinarySerializer as HDF5BinarySerializer


def initialize_data_serializer(
    parameters: Parameters,
) -> DataSerializerProtocol:
    """
    Initializes the Data Serializer specified by the configuration parameters

    Arguments:

        parameters: The configuration parameters

    Returns:

        data_serializer: An initialized Data Serializer
    """
    data_serializer_parameters: DataSerializerParameters = parameters.data_serializer

    try:
        data_serializer: DataSerializerProtocol = globals()[
            data_serializer_parameters.type
        ](data_serializer_parameters)
    except NameError:
        log_error_and_exit(
            f"Data serializer {data_serializer_parameters.type} is not available"
        )

    return data_serializer
