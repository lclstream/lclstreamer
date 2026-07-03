import traceback

from click.testing import Result
from typer.testing import CliRunner

from lclstreamer.cmd.lclstreamer import app

runner: CliRunner = CliRunner()


def test_app() -> None:
    result: Result = runner.invoke(app, ["--help"])
    print("--- Output")
    print(result.output)
    if result.exception is not None and result.exc_info is not None:
        print("--- Exceptions")
        print(result.exception)
        traceback.print_tb(result.exc_info[2])

    assert result.exit_code == 0
