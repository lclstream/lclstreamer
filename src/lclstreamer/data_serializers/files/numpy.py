from collections.abc import Iterator
from io import BytesIO

import numpy

from ...models.parameters import (
    NumpyBinarySerializerParameters,
)
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSerializerProtocol
from ...utils.typing import StrFloatIntNDArray


class NumpyBinarySerializer(DataSerializerProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(self, parameters: NumpyBinarySerializerParameters) -> None:
        """
        Initializes a Numpy data serializer

        This serializer turns a dictionary of numpy arrays into a binary blob with the
        internal structure of a .npz file (numpy's native format), according to the
        preferences specified by the configuration parameters.

        Arguments:

            parameters: The configuration parameters
        """
        if parameters.type != "NumpyBinarySerializer":
            log_error_and_exit(
                "Data serializer parameters do not match the expected type"
            )

        self._use_compression: bool = parameters.use_compression

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray | None]]
    ) -> Iterator[bytes]:
        """
        Serializes a stream of event data dictionaries to .npz binary blobs

        Arguments:

            stream: An iterator of event data dictionaries

        Yields:

            byte_block: A binary blob (a bytes object) with .npz internal structure
        """
        data: dict[str, StrFloatIntNDArray | None]
        for data in stream:
            # Filter out None values before serialization
            arrays: dict[str, StrFloatIntNDArray] = {
                k: v for k, v in data.items() if v is not None
            }

            with BytesIO() as byte_block:
                if self._use_compression:
                    numpy.savez_compressed(byte_block, **arrays)
                else:
                    numpy.savez(byte_block, **arrays)

                yield byte_block.getvalue()
