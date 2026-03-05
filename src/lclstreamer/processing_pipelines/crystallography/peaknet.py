from collections.abc import Iterator
from typing import Any, Literal

import numpy
from numpy.typing import NDArray

from ...models.parameters import (
    PeaknetPreprocessingPipelineParameters,
)
from ...utils.logging import log_error_and_exit
from ...utils.protocols import ProcessingPipelineProtocol
from ...utils.typing import StrFloatIntNDArray
from ..common.data_storage import DataStorage


class _NumpyPad:
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self,
        target_height: int,
        target_width: int,
        pad_style: Literal["center", "bottom-right"] = "center",
    ):
        """
        Initializes a numpy array padder to a specified target size

        This class handles padding of 3D numpy arrays (C, H, W) to target dimensions.
        Only the spatial dimensions (H, W) are padded; the channel dimension is
        unchanged. It supports different padding styles and uses zero-padding

        Arguments:

            target_height: Target height after padding

            target_width: Target width after padding

            pad_style: Padding style - either "center" or "bottom-right"
        """
        self.target_height = target_height
        self.target_width = target_width
        self.pad_style = pad_style

    def calc_pad_width(
        self, img: NDArray[numpy.floating[Any]]
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Calculates padding widths for the given image

        Arguments:

            img: Input image array with shape (C, H, W)

        Returns:

            pad_width: Padding widths in format ((0, 0), (top, bottom), (left, right))
        """
        C_img, H_img, W_img = img.shape

        dH_padded = max(self.target_height - H_img, 0)
        dW_padded = max(self.target_width - W_img, 0)

        if self.pad_style == "center":
            pad_width = (
                (0, 0),  # No padding on channel dimension
                (dH_padded // 2, dH_padded - dH_padded // 2),  # (top, bottom)
                (dW_padded // 2, dW_padded - dW_padded // 2),  # (left, right)
            )
        elif self.pad_style == "bottom-right":
            pad_width = (
                (0, 0),  # No padding on channel dimension
                (0, dH_padded),  # (top, bottom)
                (0, dW_padded),  # (left, right)
            )
        else:
            raise ValueError(
                f"Invalid pad_style '{self.pad_style}'. Use 'center' or 'bottom-right'."
            )

        return pad_width

    def __call__(
        self,
        img: NDArray[numpy.floating[Any]],
    ) -> NDArray[numpy.floating[Any]]:
        """
        Applies padding to the input image

        Arguments:

            img: Input image array with shape (C, H, W)

        Returns:

            img_padded: Padded image array with shape (C, target_height, target_width)
        """
        pad_width = self.calc_pad_width(img)
        img_padded: NDArray[numpy.floating[Any]] = numpy.pad(
            img, pad_width, mode="constant", constant_values=0
        )

        return img_padded


class _NumpyInstanceNorm:
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self,
        eps: float = 1e-5,
        check_nan: bool = True,
        scale_variance: bool = True,
    ):
        """
        Initializes an instance normalizer for numpy arrays

        Normalizes each sample independently across all dimensions. Adapted from
        peaknet's InstanceNorm for numpy compatibility

        Arguments:

            eps: Small constant for numerical stability in variance computation

            check_nan: Whether to check for NaN values in normalized output

            scale_variance: Whether to scale by standard deviation (if False,
                only centers the data)
        """
        self.eps = eps
        self.check_nan = check_nan
        self.scale_variance = scale_variance

    def __call__(
        self, img: NDArray[numpy.floating[Any]]
    ) -> NDArray[numpy.floating[Any]]:
        """
        Applies instance normalization to a single image

        Arguments:

            img: Input image array with shape (C, H, W)

        Returns:

            img_norm: Normalized image array with same shape (C, H, W)

        Raises:

            ValueError: If NaN values are detected in output and check_nan is True
        """
        mean = numpy.mean(img)
        img_norm: NDArray[numpy.floating[Any]] = img - mean

        if self.scale_variance:
            variance = numpy.var(img, ddof=0)
            img_norm = img_norm / numpy.sqrt(variance + self.eps)

        if self.check_nan and numpy.isnan(img_norm).any():
            raise ValueError("NaN values detected in normalized output")

        return img_norm


def _add_channel_dimension(image_array: StrFloatIntNDArray) -> StrFloatIntNDArray:
    """
    Adds a channel dimension to an individual image

    Converts (H, W) format to (1, H, W) format by adding a channel dimension

    Arguments:

        image_array: Input image array with shape (H, W)

    Returns:

        result: Array with added channel dimension (1, H, W)

    Raises:

        ValueError: If input array doesn't have expected 2D shape
    """
    if len(image_array.shape) != 2:
        raise ValueError(
            f"Expected 2D input array (H, W), got shape {image_array.shape}"
        )

    result: StrFloatIntNDArray = numpy.expand_dims(image_array, axis=0)
    return result


def _is_image_data(data_key: str, data_value: StrFloatIntNDArray | None) -> bool:
    # Determines if a data entry represents image data that should be preprocessed
    # Heuristic: 2D arrays (H, W) or 3D arrays (C, H, W) are likely images
    return isinstance(data_value, numpy.ndarray) and len(data_value.shape) in [2, 3]


class PeaknetPreprocessingPipeline(ProcessingPipelineProtocol):
    """
    See documentation of the `__init__` function
    """

    def __init__(self, parameters: PeaknetPreprocessingPipelineParameters) -> None:
        """
        Initializes a PeakNet Preprocessing Pipeline

        This pipeline applies image preprocessing and batching in the following
        order:

        1. Add channel dimension to individual images if needed: (H, W) -> (1, H, W)
        2. Pad individual images to the configured target size: (C, H, W) ->
           (C, target_height, target_width)
        3. Apply instance normalization if enabled (before batching)
        4. Accumulate preprocessed images into batches of ``batch_size``
        5. Optionally merge batch and channel dimensions: (B, C, H, W) ->
           (B*C, 1, H, W)

        Non-image data (timestamps, scalars, etc.) is passed through unchanged
        at every stage

        Arguments:

            parameters: The processing pipeline configuration parameters
        """

        if parameters.type != "PeaknetPreprocessingPipeline":
            log_error_and_exit(
                "Processing pipeline parameters do not match the expected type"
            )

        self._batch_size: int = parameters.batch_size

        # Initialize padding utility
        self._padder: _NumpyPad = _NumpyPad(
            target_height=parameters.target_height,
            target_width=parameters.target_width,
            pad_style=parameters.pad_style,
        )

        # Channel dimension settings
        self._add_channel_dim: bool = parameters.add_channel_dim
        self._merge_batch_and_channel: bool = parameters.merge_batch_and_channel

        # Normalization settings
        self._apply_normalization: bool = parameters.apply_normalization
        if self._apply_normalization:
            self._normalizer: _NumpyInstanceNorm = _NumpyInstanceNorm(
                eps=parameters.norm_eps,
                check_nan=parameters.norm_check_nan,
                scale_variance=parameters.norm_scale_variance,
            )

    def _apply_batch_channel_merge(
        self, batched_data: dict[str, StrFloatIntNDArray | None]
    ) -> None:
        """
        Reshapes image data from (B, C, H, W) to (B*C, 1, H, W) if enabled

        Arguments:

            batched_data: Dictionary of batched arrays to modify in-place
        """
        if not self._merge_batch_and_channel:
            return

        data_key: str
        data_value: StrFloatIntNDArray | None
        for data_key, data_value in batched_data.items():
            if data_value is None or data_key.endswith("_original_shape"):
                continue

            if len(data_value.shape) == 4:
                B, C, H, W = data_value.shape
                batched_data[data_key] = data_value.reshape(B * C, 1, H, W)

    def _finalize_batch(
        self,
        data_storage: DataStorage,
        original_shapes: dict[str, tuple[int, int, int]],
    ) -> dict[str, StrFloatIntNDArray | None]:
        """
        Retrieves stored data, adds shape metadata, and applies batch-channel merge

        Arguments:

            data_storage: The data storage containing accumulated events

            original_shapes: Dictionary mapping data keys to their original
                (C, H, W) shapes before preprocessing

        Returns:

            batched_data: The finalized batch dictionary
        """
        batched_data: dict[str, StrFloatIntNDArray | None] = (
            data_storage.retrieve_stored_data()
        )

        # Add original shape metadata for each image data key
        data_key: str
        for data_key, (orig_c, orig_h, orig_w) in original_shapes.items():
            if batched_data.get(data_key) is not None:
                batch_size: int = batched_data[data_key].shape[0]  # type: ignore[union-attr]
                batched_data[f"{data_key}_original_shape"] = numpy.array(
                    [batch_size, orig_c, orig_h, orig_w], dtype=numpy.int32
                )

        self._apply_batch_channel_merge(batched_data)

        return batched_data

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray | None]]
    ) -> Iterator[dict[str, StrFloatIntNDArray | None]]:
        """
        Applies the PeakNet Preprocessing Pipeline to incoming event data

        For each event, the steps of the processing pipeline are applied to the
        incoming data. Any remaining events that do not fill a complete batch at
        the end of the data stream are yielded as a partial batch.

        Arguments:

            stream: An iterator of event data dictionaries

        Yields:

            batch: A dictionary of processed and batched events
        """
        data_storage: DataStorage = DataStorage()
        original_shapes: dict[str, tuple[int, int, int]] = {}

        data: dict[str, StrFloatIntNDArray | None]
        for data in stream:
            preprocessed_data: dict[str, StrFloatIntNDArray | None] = {}

            data_key: str
            data_value: StrFloatIntNDArray | None
            for data_key, data_value in data.items():
                if data_value is not None and _is_image_data(data_key, data_value):
                    # Capture original shape before any preprocessing (first image only)
                    if data_key not in original_shapes:
                        if len(data_value.shape) == 2:
                            original_shapes[data_key] = (
                                1,
                                data_value.shape[0],
                                data_value.shape[1],
                            )
                        elif len(data_value.shape) == 3:
                            original_shapes[data_key] = (
                                data_value.shape[0],
                                data_value.shape[1],
                                data_value.shape[2],
                            )

                    # Add channel dimension if needed: (H, W) -> (1, H, W)
                    if self._add_channel_dim and len(data_value.shape) == 2:
                        data_value = _add_channel_dimension(data_value)

                    # Apply padding: (C, H, W) -> (C, target_height, target_width)
                    data_value = self._padder(data_value)

                    # Apply normalization before batching (if enabled)
                    if self._apply_normalization:
                        data_value = self._normalizer(data_value)

                    preprocessed_data[data_key] = data_value
                else:
                    preprocessed_data[data_key] = data_value

            data_storage.add_data(data=preprocessed_data)

            if len(data_storage) >= self._batch_size:
                yield self._finalize_batch(data_storage, original_shapes)
                data_storage.reset_data_storage()
                original_shapes = {}

        # Handle remaining data
        if len(data_storage) > 0:
            yield self._finalize_batch(data_storage, original_shapes)
