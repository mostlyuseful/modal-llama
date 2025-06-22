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
