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

from pathlib import Path
from modal_llama.llama_swap import LlamaSwapModel
import shlex
import logging

logger = logging.getLogger(__name__)


class LlamaCppConfig:
    def __init__(
        self,
        name: str,
        binary: str | Path,
        model: str | Path,
        params: dict | None = None,
    ):
        self.name = name
        self.binary = Path(binary)
        self.model = Path(model)
        self._params = params or {}

    def with_params(self, **params: str | float | bool | Path) -> "LlamaCppConfig":
        """
        Return a new LlamaCppConfig with the specified parameters.
        """
        new_params = self._params.copy()
        new_params.update(params)
        return LlamaCppConfig(
            name=self.name, binary=self.binary, model=self.model, params=new_params
        )

    def without_params(self, *param_names: str) -> "LlamaCppConfig":
        """
        Return a new LlamaCppConfig without the specified parameters.
        """
        new_params = {k: v for k, v in self._params.items() if k not in param_names}
        return LlamaCppConfig(
            name=self.name, binary=self.binary, model=self.model, params=new_params
        )

    def build(self) -> LlamaSwapModel:
        """
        Build the LlamaCppConfig into a LlamaSwapModel.

        Validates config, escapes file paths and builds commandline (parameters/flags)
        """

        if not self.binary.exists():
            raise ValueError(f"Binary {self.binary} does not exist.")
        if not self.model.exists():
            raise ValueError(f"Model {self.model} does not exist.")

        quoted_cmdline = (
            shlex.quote(str(self.binary)) + " -m " + shlex.quote(str(self.model))
        )
        for k, v in self._params.items():
            k = k.replace(
                "_", "-"
            )  # Replace underscores with hyphens for command line flags
            if v is None:
                quoted_cmdline += f" --{k}"
            elif isinstance(v, bool):
                if v:
                    quoted_cmdline += f" --{k}"
            else:
                if isinstance(v, Path):
                    v = str(v.resolve())
                    quoted_cmdline += f" --{k} {shlex.quote(v)}"
                else:
                    v = str(v)
                    quoted_cmdline += f" --{k} {v}"

        logger.debug(
            f"Building LlamaSwapModel for {self.name} with command: {quoted_cmdline}"
        )

        return LlamaSwapModel(
            name=self.name,
            cmd=quoted_cmdline,
        )
