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

from modal import Image
from dataclasses import dataclass


@dataclass
class BuildInfo:
    """Dataclass containing the llama.cpp build's binary directory and image."""

    bin_dir: str
    image: Image


def build_llama_cpp(image: Image) -> BuildInfo:
    """
    Build llama.cpp with CUDA support.

    This function:
    1. Clones the llama.cpp repository
    2. Builds it with CUDA support using CMake and Ninja
    3. Returns a dataclass with the deployment and image for chaining
    """
    # Create a stub directory for the llama.cpp build
    LLAMA_CPP_DIR = "/opt/llama.cpp"

    image = image.apt_install(
        "ninja-build", "cmake", "gcc", "g++", "git", "libcurl4-openssl-dev"
    )
    image = image.env(
        {"LD_LIBRARY_PATH": "/usr/local/cuda-12.8/compat/:$LD_LIBRARY_PATH"}
    )
    image = image.run_commands(
        [
            "git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
            "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release",
            "cmake --build /opt/llama.cpp/build --config Release",
        ]
    )

    return BuildInfo(bin_dir=LLAMA_CPP_DIR + "/build/bin", image=image)


def build_ik_llama_cpp(image: Image) -> BuildInfo:
    """
    Build ik_llama.cpp with CUDA support.

    This function:
    1. Clones the ik_llama.cpp repository
    2. Builds it with CUDA support using CMake and Ninja
    3. Returns a dataclass with the deployment and image for chaining
    """
    # Create a stub directory for the ik_llama.cpp build
    IK_LLAMA_CPP_DIR = "/opt/ik_llama.cpp"

    image = image.apt_install(
        "ninja-build", "cmake", "gcc", "g++", "git", "libcurl4-openssl-dev"
    )
    image = image.env(
        {"LD_LIBRARY_PATH": "/usr/local/cuda-12.8/compat/:$LD_LIBRARY_PATH"}
    )
    image = image.run_commands(
        [
            "git clone https://github.com/ikawrakow/ik_llama.cpp.git /opt/ik_llama.cpp",
            "cmake -S /opt/ik_llama.cpp -B /opt/ik_llama.cpp/build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release",
            "cmake --build /opt/ik_llama.cpp/build --config Release --target llama-server",
        ]
    )

    return BuildInfo(bin_dir=IK_LLAMA_CPP_DIR + "/build/bin", image=image)


def build_llama_swap(image: Image) -> BuildInfo:
    """
    Build llama_swap

    This function:
    1. Clones the llama_swap repository
    2. Builds it using Go
    3. Returns a dataclass with the deployment and image for chaining
    """
    # Create a stub directory for the llama_swap build
    LLAMA_SWAP_DIR = "/opt/llama_swap"

    image = image.apt_install("git", "golang-go", "npm")
    image = image.run_commands(
        [
            "git clone https://github.com/mostlygeek/llama-swap.git /opt/llama_swap",
            "cd /opt/llama_swap && make linux",
        ]
    )

    return BuildInfo(bin_dir=LLAMA_SWAP_DIR + "/build", image=image)
