import os
from pathlib import Path
import typer

from modal_llama.llama_swap import LlamaSwapConfig, start_llama_swap_server
from modal_llama.models import prep_common_models
from modal_llama.nginx import start_nginx_reverse_proxy

app = typer.Typer()


def get_config(
    models_cache_path: str | Path,
    llama_cpp_backend_path: str | Path,
    ik_llama_cpp_backend_path: str | Path,
    llama_swap_listen_port: int = 8080,
):
    # Monkey patch for huggingface_hub to use the correct cache directory
    # This is necessary to ensure that the models are downloaded by gguf()->snapshot_download() to the correct location.
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_CACHE = str(models_cache_path)

    cfg = LlamaSwapConfig(listen_port=llama_swap_listen_port)

    print("Adding models to configuration and downloading them...")
    prep_common_models(
        cfg,
        llama_cpp_backend=str(llama_cpp_backend_path),
        ik_llama_cpp_backend=str(ik_llama_cpp_backend_path),
    )

    return cfg


@app.command()
def serve(
    models_cache_path: str = typer.Option(
        ..., help="Path to the models cache directory, defaults to /models"
    ),
    llama_cpp_backend_path: str = typer.Option(
        ...,
        help="Path to the llama.cpp backend binary, e.g. /opt/llama-cpp/bin/llama-server",
    ),
    ik_llama_cpp_backend_path: str = typer.Option(
        ...,
        help="Path to the ik_llama.cpp backend binary, e.g. /opt/ik-llama-cpp/bin/llama-server",
    ),
    llama_swap_bin_dir: str = typer.Option(
        ...,
        help="Path to the directory containing the llama-swap binary, e.g. /opt/llama-swap/build",
    ),
    llama_swap_listen_port: int = typer.Option(
        8080, help="Port for the llama-swap server, defaults to 8080"
    ),
    nginx_port: int = typer.Option(
        8000,
        help="Port for the nginx reverse proxy, defaults to 8000. This port will have to be open in the firewall for external access.",
    ),
    detach: bool = typer.Option(
        False,
        help="Run the server in detached mode, i.e. run it in the background and exit immediately. Needed for modal deployment.",
    ),
):
    print("Downloading models...")
    # Side-effect of this function is to download the models to the specified cache directory.
    cfg = get_config(
        models_cache_path,
        llama_cpp_backend_path=llama_cpp_backend_path,
        ik_llama_cpp_backend_path=ik_llama_cpp_backend_path,
        llama_swap_listen_port=llama_swap_listen_port,
    )
    print("Starting llama-swap server...")
    api_token = os.environ.get("API_TOKEN", None)
    if api_token is None:
        print("API_TOKEN is not set, application is not protected by token authentication!")
        print("You can set it by creating a .env file with the line: API_TOKEN=your_token_here")
    nginx_proc, _ = start_nginx_reverse_proxy(api_token=api_token, llama_swap_port=llama_swap_listen_port, listen_port=nginx_port)
    llama_swap_proc = start_llama_swap_server(llama_swap_bin_dir, cfg)
    if detach:
        print("Running in detached mode, exiting immediately...")
        return

    # Wait on both processes
    try:
        print("Llama-swap server is running. Press Ctrl+C to stop.")
        llama_swap_proc.wait()
    except KeyboardInterrupt:
        print("Stopping llama-swap server...")
        llama_swap_proc.terminate()
        print("Stopping nginx reverse proxy...")
        nginx_proc.terminate()
    print("Exiting...")


if __name__ == "__main__":
    app()