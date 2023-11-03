"""The custom errors raised by the model server."""
import typing


class ModelServerException(Exception):
    """The base class for any custom expections."""


class UnsupportedFormatException(ModelServerException):
    """An error that indicates the model format is not supported for the provided type."""

    def __init__(self, model_type: str, supported: typing.List[str]):
        """Initialize the exception."""
        super().__init__(
            "Unsupported model type and format combination. "
            + f"{model_type} models are supported in the following formats: {str(supported)}"
        )
