# This file is part of Modal Llama.
# Modal Llama is a wrapper around llama.cpp, ik_llama.cpp, llama-swap and nginx to run them on Modal and other cloud providers.
# Copyright (C) 2025  Maurice-Pascal Sonnemann (mpsonnemann@gmail.com)
#
# Modal Llama is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Modal Llama is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The source code is available at https://github.com/mostlyuseful/modal-llama

import typer
import runpod
from runpod.api.ctl_commands import get_gpus

app = typer.Typer()


@app.command()
def list_gpus(
    api_key: str = typer.Option(
        ..., help="Runpod API key for authentication", envvar="RUNPOD_API_KEY"
    ),
):
    """
    Check GPU availability in runpod ecosystem.
    """
    runpod.api_key = api_key
    gpus = get_gpus()
    if gpus:
        typer.echo("Available GPUs:")
        for gpu in gpus:
            typer.echo(f" - {gpu}")
    else:
        typer.echo("No GPUs available.")


if __name__ == "__main__":
    app()
