# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A wrapper on PandasAI base LLM class to use NVIDIA Foundational Models in PandasAI Agents"""

from typing import Any, Dict, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pandasai.llm.base import LLM
from pandasai.prompts.base import AbstractPrompt


class NVIDIA(LLM):
    """
    A wrapper class on PandasAI base LLM class to NVIDIA Foundational Models.
    """

    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1
    model: Optional[str] = "llama2_13b"

    _chat_model: "ChatNVIDIA" = None

    def __init__(self, **kwargs):
        self._set_params(**kwargs)
        self._chat_model = ChatNVIDIA(**self._default_params)
        self._prompt = ""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling NVIDIA Foundational LLMs."""
        params: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        return params

    @property
    def type(self) -> str:
        return "nvidia-foundational-llm"

    def _set_params(self, **kwargs):
        """
        Set Parameters
        Args:
            **kwargs: ["model","temperature","max_tokens",
            "top_p"]

        Returns:
            None.
        """

        valid_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    def call(self, instruction: AbstractPrompt, suffix: str = "") -> str:
        """
        Call the NVIDIA Foundational LLMs.
        Args:
            instruction (AbstractPrompt): A prompt object with instruction for LLM.
            suffix (str): A string representing the suffix to be truncated
                from the generated response.

        Returns
            str: LLM response.

        """
        self._prompt = instruction.to_string().replace("`", "'") + suffix
        response = self._chat_model.invoke(self._prompt)
        return response.content
