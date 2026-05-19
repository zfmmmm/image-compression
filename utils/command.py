from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess
from collections.abc import Mapping, Sequence


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult):
        command = " ".join(result.args)
        message = (
            f"Command failed with return code {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        super().__init__(message)
        self.result = result


def run_command(
    args: Sequence[str | Path],
    *,
    timeout: float = 600.0,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run a subprocess, capturing stdout/stderr and checking return code."""

    text_args = [str(arg) for arg in args]
    LOGGER.debug("Running command: %s", " ".join(text_args))
    completed = subprocess.run(
        text_args,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    result = CommandResult(
        args=text_args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if result.returncode != 0:
        raise CommandError(result)
    return result
