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

# Adapted from example at https://modal.com/blog/how_to_run_deepseek_r1_distilled_vllm

from dataclasses import dataclass
from datetime import timedelta
import os

from modal import App, Image, Secret, Volume, concurrent, web_server, enable_output

from modal_llama.modal.build import (
    build_ik_llama_cpp,
    build_llama_cpp,
    build_llama_swap,
)
from modal_llama.llama_swap import LlamaSwapConfig, start_llama_swap_server
from modal_llama.models import prep_common_models
from modal_llama.nginx import start_nginx_reverse_proxy

image = (
    Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .pip_install("uv")
    .apt_install("ninja-build", "cmake", "gcc", "g++", "git")
)
llama_cpp_build = build_llama_cpp(image)
ik_llama_cpp_build = build_ik_llama_cpp(llama_cpp_build.image)
llama_swap_build = build_llama_swap(ik_llama_cpp_build.image)
final_image = (
    llama_swap_build.image
    # .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    # .copy_local_dir(HERE, "/src/app", ignore=["__pycache__", "*.wav"])
    # .copy_local_file(PROJECT_ROOT / "pyproject.toml", "/")
    # .copy_local_file(PROJECT_ROOT / "uv.lock", "/")
    # .pip_install("uv")
    # .run_commands("uv sync --frozen --no-dev --no-editable")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("nginx")
    .add_local_python_source("modal_llama")
)

MODELS_PATH = "/models"
models_volume = Volume.from_name("llama-models", create_if_missing=True)

app = App("modal-llama", image=final_image)


@dataclass(frozen=True)
class Environment:
    llama_cpp_backend: str
    ik_llama_cpp_backend: str
    cfg: LlamaSwapConfig


@app.function(
    volumes={MODELS_PATH: models_volume},
    timeout=int(timedelta(minutes=30).total_seconds()),
)
def prep():
    """
    Prepare the models volume by creating the directory structure.
    This is necessary to ensure the volume is ready for use.
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Preparing models volume...")

    llama_cpp_backend = llama_cpp_build.bin_dir + "/llama-server"
    ik_llama_cpp_backend = ik_llama_cpp_build.bin_dir + "/llama-server"

    # Monkey patch for huggingface_hub to use the correct cache directory
    # This is necessary to ensure that the models are downloaded by gguf()->snapshot_download() to the correct location.
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_CACHE = MODELS_PATH
    # huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True

    cfg = LlamaSwapConfig(listen_port=8080)

    print("Adding models to configuration and downloading them...")
    prep_common_models(
        cfg,
        llama_cpp_backend=llama_cpp_backend,
        ik_llama_cpp_backend=ik_llama_cpp_backend,
    )

    return Environment(
        llama_cpp_backend=llama_cpp_backend,
        ik_llama_cpp_backend=ik_llama_cpp_backend,
        cfg=cfg,
    )


@app.function(
    gpu="H200",
    scaledown_window=int(timedelta(minutes=10).total_seconds()),
    min_containers=0,
    max_containers=1,
    volumes={MODELS_PATH: models_volume},
    timeout=int(timedelta(minutes=10).total_seconds()),
    secrets=[Secret.from_dotenv()],
)
@concurrent(max_inputs=100)
@web_server(
    port=8000,
    startup_timeout=int(timedelta(minutes=10).total_seconds()),
)
def serve():
    env = prep.local()
    nginx_port = 8000

    api_token = os.environ.get("API_TOKEN", None)
    if api_token is None:
        print(
            "API_TOKEN is not set, application is not protected by token authentication!"
        )
        print(
            "You can set it by creating a .env file with the line: API_TOKEN=your_token_here"
        )

    # Start the LlamaSwap server in detached mode
    # This will run in the background and allow the app to continue running
    # It's required so modal can manage the lifecycle of the server
    # even if this feels a bit hacky :)
    start_nginx_reverse_proxy(
        api_token=api_token, llama_swap_port=env.cfg.listen_port, listen_port=nginx_port
    )
    start_llama_swap_server(llama_swap_build.bin_dir, env.cfg)


def deploy():
    with enable_output():
        with app.run():
            input("Press <ENTER> to stop the server...")


if __name__ == "__main__":
    deploy()
