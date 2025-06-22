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

from datetime import timedelta
import os.path
from pathlib import Path
import subprocess
from dataclasses import dataclass
import yaml


@dataclass
class LlamaSwapModel:
    name: str
    cmd: str
    aliases: list[str] | None = None
    ttl: timedelta | None = None
    check_endpoint: str | None = None
    env: dict[str, str] | None = None
    unlisted: bool | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "cmd": self.cmd,
        }
        if self.aliases is not None:
            d["aliases"] = self.aliases
        if self.ttl is not None:
            d["ttl"] = int(self.ttl.total_seconds())
        if self.check_endpoint is not None:
            d["checkEndpoint"] = self.check_endpoint
        if self.env is not None:
            d["env"] = [f"{k}={v}" for k, v in self.env.items()]
        if self.unlisted is not None:
            d["unlisted"] = self.unlisted
        return d


class LlamaSwapConfig:
    def __init__(
        self,
        listen_port: int,
        health_check_timeout: timedelta = timedelta(minutes=5),
        log_level: str = "debug",
    ):
        self.listen_port = listen_port
        self.health_check_timeout = health_check_timeout
        self.log_level = log_level
        self.models = {}

    def add_model(self, model: LlamaSwapModel):
        self.models[model.name] = model
        return self

    def to_yaml(self) -> str:
        payload = {
            "healthCheckTimeout": int(self.health_check_timeout.total_seconds()),
            "logLevel": self.log_level,
            "models": {name: model.to_dict() for name, model in self.models.items()},
        }
        return yaml.dump(payload)


def start_llama_swap_server(
    bin_dir: str | Path, cfg: LlamaSwapConfig
) -> subprocess.Popen:
    """
    Start the LlamaSwap server with the given configuration.
    """
    with open("/tmp/llama_swap_config.yaml", "w") as f:
        f.write(cfg.to_yaml())

    with open("/tmp/llama_swap_config.yaml", "r") as f:
        print(f"\n\n\n=== Using LlamaSwap config ===\n{f.read()}\n\n\n")

    return subprocess.Popen(
        [
            os.path.join(bin_dir, "llama-swap-linux-amd64"),
            "-config",
            "/tmp/llama_swap_config.yaml",
            "-listen",
            f"0.0.0.0:{cfg.listen_port}",
        ],
        stderr=None,  # Redirect stderr to the default (console)
        stdout=None,  # Redirect stdout to the default (console)
    )
