import logging
import sys
from typing import Any, NoReturn

from rich.logging import RichHandler


def log_error_and_exit(error: str) -> NoReturn:
    """
    Logs an error message and exits the program

    Arguments:

        error: The error message to log before exiting
    """
    log.error(error)
    sys.exit(1)

def log_error(error: str) -> None:
    """
    Logs an error message

    Arguments:

        error: The error message to log
    """
    log.error(error)

def log_info(info: str) -> None:
    """
    Logs an informational message

    Arguments:

        info: The informational message to log
    """
    log.info(info)

def log_debug(debugmsg: str) -> None:
    """
    Logs a debug message

    Arguments:

        debugmsg: The debug message to log
    """
    log.debug(debugmsg)


class RichHandlerWithAggregation(RichHandler):
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self, *, recurring_msg_emit_interval: int = 100, **kwargs: Any
    ) -> None:
        """
        Initializes a Rich log handler with message aggregation

        This handler suppresses repeated identical log messages, only re-emitting
        them periodically

        Arguments:

            recurring_msg_emit_interval: The number of repetitions between
                re-emissions of a recurring message. Defaults to 100

            **kwargs: Additional keyword arguments forwarded to `RichHandler`
        """
        super().__init__(**kwargs)
        self._recurring_msg_emit_interval = recurring_msg_emit_interval
        self._recurring_msg: str = ""
        self._last_recurring_record: logging.LogRecord | None = None
        self._recurring_msg_counter: int = 0

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record, appending a repetition count for recurring messages

        Arguments:

            record: The log record to format

        Returns:

            formatted: The formatted log record string
        """
        if self._recurring_msg_counter != 0:
            record.msg = f"{record.msg} (Repeated {self._recurring_msg_counter} times)"
        return super().format(record)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emits a log record, suppressing duplicates and re-emitting them periodically

        If the incoming record has the same message as the previous one, the emission
        is suppressed and an internal counter is incremented. Once the counter reaches
        a multiple of `recurring_msg_emit_interval`, the record is re-emitted with the
        repetition count appended. When the message changes, any accumulated repetition
        count for the previous message is flushed first

        Arguments:

            record: The log record to emit
        """
        if record.msg != self._recurring_msg:
            if (
                self._recurring_msg_counter != 0
                and self._last_recurring_record is not None
            ):
                super().emit(self._last_recurring_record)
            self._last_recurring_record = record
            self._recurring_msg = record.msg
            self._recurring_msg_counter = 0
            super().emit(record)
        else:
            self._recurring_msg_counter += 1
            self._last_recurring_record = record
            if self._recurring_msg_counter % self._recurring_msg_emit_interval == 0:
                super().emit(self._last_recurring_record)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandlerWithAggregation(rich_tracebacks=True, show_path=False)],
)

log: logging.Logger = logging.getLogger("rich")
