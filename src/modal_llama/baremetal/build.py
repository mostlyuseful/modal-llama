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

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import typer


def get_cpu_count() -> int:
    """
    Get the number of CPU cores available on the system.
    """
    try:
        return int(subprocess.check_output(["nproc"]).strip())
    except subprocess.CalledProcessError:
        return 8  # Fallback if nproc fails


@dataclass
class BuildInfo:
    """Dataclass containing the llama.cpp build's binary directory."""

    bin_dir: Path


def build_llama_cpp(repo_dir: str | Path) -> BuildInfo:
    """
    Build llama.cpp with CUDA support.

    This function:
    1. Clones the llama.cpp repository
    2. Builds it with CUDA support using CMake and Ninja
    3. Returns a dataclass with the deployment and image for chaining
    """
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "ninja-build",
            "cmake",
            "gcc",
            "g++",
            "git",
            "libcurl4-openssl-dev",
        ],
        check=True,
        env={"DEBIAN_FRONTEND": "noninteractive"},
    )

    if Path(repo_dir).exists():
        typer.echo(
            f"Directory {repo_dir} already exists. Skipping clone, pulling instead."
        )
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ggml-org/llama.cpp.git",
                str(repo_dir),
            ],
            check=True,
        )

    print("Running CMake")
    subprocess.run(
        f"PATH=$PATH:/usr/local/cuda/bin cmake -S {repo_dir} -B {Path(repo_dir) / 'build'} -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release",
        shell=True,
        check=True,
        executable="/bin/bash",
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(Path(repo_dir) / "build"),
            "-j",
            str(min(16, get_cpu_count())),
            "--config",
            "Release",
        ],
        check=True,
    )

    return BuildInfo(bin_dir=Path(repo_dir) / "build" / "bin")


def build_ik_llama_cpp(repo_dir: str | Path) -> BuildInfo:
    """
    Build ik_llama.cpp with CUDA support.

    This function:
    1. Clones the ik_llama.cpp repository
    2. Builds it with CUDA support using CMake and Ninja
    3. Returns a dataclass with the deployment and image for chaining
    """
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "ninja-build",
            "cmake",
            "gcc",
            "g++",
            "git",
            "libcurl4-openssl-dev",
        ],
        check=True,
        env={"DEBIAN_FRONTEND": "noninteractive"},
    )

    if Path(repo_dir).exists():
        typer.echo(
            f"Directory {repo_dir} already exists. Skipping clone, pulling instead."
        )
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ikawrakow/ik_llama.cpp.git",
                str(repo_dir),
            ],
            check=True,
        )

    subprocess.run(
        f"PATH=$PATH:/usr/local/cuda/bin cmake -S {repo_dir} -B {Path(repo_dir) / 'build'} -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release",
        shell=True,
        check=True,
        executable="/bin/bash",
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(Path(repo_dir) / "build"),
            "--config",
            "Release",
            "-j",
            str(min(16, get_cpu_count())),
            "--target",
            "llama-server",
        ],
        check=True,
    )
    return BuildInfo(bin_dir=Path(repo_dir) / "build" / "bin")


def ensure_nodejs():
    """
    Ensure Node.js >18 is installed on the system.
    If not, install it using the NodeSource setup script.
    """
    is_nodejs_present = False
    try:
        version = subprocess.check_output(["node", "--version"]).strip().decode("utf-8")
        major = int(version.split(".")[0][1:])  # Extract major version from "v18.16.0"
        if major >= 18:
            is_nodejs_present = True
            typer.echo(f"Node.js version {version} is already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo("Node.js is not installed or not found in PATH.")

    if not is_nodejs_present:
        typer.echo("Installing Node.js 20.x using NodeSource setup script...")
        subprocess.run(
            "curl -fsSL https://deb.nodesource.com/setup_20.x | bash",
            shell=True,
            check=True,
        )
        # First, uninstall any existing Node.js versions: nodejs, npm and libnode*
        subprocess.run(
            ["apt-get", "remove", "-y", "nodejs", "npm", "libnode*"], check=True
        )
        # Then, install upgraded Node.js
        subprocess.run(["apt-get", "install", "-y", "nodejs"], check=True)
        installed_version = (
            subprocess.check_output(["node", "--version"]).strip().decode("utf-8")
        )
        typer.echo(f"Node.js {installed_version} installed successfully.")


def ensure_go():
    """
    Ensure up-to-date Go is installed on the system.
    If not, install it using the official Go installation script.
    """
    is_go_present = False
    try:
        version = subprocess.check_output(["go", "version"]).strip().decode("utf-8")
        if "go version go" in version:
            match = re.search(r"go version go(\d+)\.(\d+)\.(\d+)", version)
            if match:
                major, minor, patch = map(int, match.groups())
                if major >= 1 and minor >= 23 and patch >= 0:
                    is_go_present = True
                    typer.echo(f"Go is already installed: {major}.{minor}.{patch}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo("Go is not installed or not found in PATH.")

    if not is_go_present:
        typer.echo("Installing Go...")
        subprocess.run(
            "curl -fsSL https://go.dev/dl/go1.24.4.linux-amd64.tar.gz | tar -xz -C /usr/local",
            shell=True,
            check=True,
        )
        subprocess.run(["export", "PATH=$PATH:/usr/local/go/bin"], shell=True)
        typer.echo("Go installed successfully.")


def build_llama_swap(repo_dir: str | Path) -> BuildInfo:
    """
    Build llama_swap

    This function:
    1. Clones the llama_swap repository
    2. Builds it using Go
    3. Returns a dataclass with the deployment and image for chaining
    """
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "git", "golang-1.23"],
        check=True,
        env={"DEBIAN_FRONTEND": "noninteractive"},
    )

    ensure_nodejs()

    if Path(repo_dir).exists():
        typer.echo(
            f"Directory {repo_dir} already exists. Skipping clone, pulling instead."
        )
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/mostlygeek/llama-swap.git",
                str(repo_dir),
            ],
            check=True,
        )

    subprocess.run(
        "PATH=$PATH:/usr/lib/go-1.23/bin make linux",
        cwd=str(repo_dir),
        check=True,
        shell=True,
    )

    return BuildInfo(bin_dir=Path(repo_dir) / "build")


app = typer.Typer(
    help="Build the baremetal dependencies for modal-llama, including llama.cpp, ik_llama.cpp, and llama-swap."
)


@app.command()
def build_baremetal(
    llama_cpp: str = typer.Option(
        "/opt/llama-cpp", help="Path to the llama.cpp repo directory"
    ),
    ik_llama_cpp: str = typer.Option(
        "/opt/ik-llama-cpp", help="Path to the ik_llama.cpp repo directory"
    ),
    llama_swap: str = typer.Option(
        "/opt/llama-swap", help="Path to the llama-swap repo directory"
    ),
):
    """
    Build the baremetal dependencies for modal-llama, including llama.cpp, ik_llama.cpp, and llama-swap.
    """
    typer.echo("Building llama.cpp...")
    typer.echo(f"{build_llama_cpp(llama_cpp)}")

    typer.echo("Building ik_llama.cpp...")
    typer.echo(f"{build_ik_llama_cpp(ik_llama_cpp)}")

    typer.echo("Building llama-swap...")
    typer.echo(f"{build_llama_swap(llama_swap)}")

    typer.echo("Build completed successfully.")


if __name__ == "__main__":
    app()
