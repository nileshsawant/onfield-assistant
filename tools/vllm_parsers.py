# SPDX-License-Identifier: Apache-2.0
"""vLLM parser plugin: registers `--tool-call-parser xml_tools` and
`--reasoning-parser think_tags`. tools/vllm-serve.sbatch loads it with
--tool-parser-plugin / --reasoning-parser-plugin.

xml_tools parses tool calls of the form
    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>
Streaming sends each tool call whole once its </tool_call> arrives.

think_tags splits <think>...</think> reasoning from the answer, treating
output as reasoning until </think> when the chat template already opened
<think> in the prompt.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
_CALL_RE = re.compile(re.escape(TOOL_START) + r"(.*?)" + re.escape(TOOL_END), re.S)
_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)


def _held_back(text: str, tag: str) -> int:
    """Length of the longest suffix of text that could be the start of tag."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class XmlToolParser(ToolParser):
    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._buf, self._content_sent, self._calls_sent = "", 0, 0

    def _param_types(self, func: str) -> dict[str, Any]:
        for tool in self.tools:
            fn = getattr(tool, "function", tool)
            if getattr(fn, "name", None) == func:
                params = getattr(fn, "parameters", None) or {}
                return params.get("properties", {}) if isinstance(params, dict) else {}
        return {}

    @staticmethod
    def _convert(raw: str, schema: Any) -> Any:
        value = raw[1:] if raw.startswith("\n") else raw
        value = value[:-1] if value.endswith("\n") else value
        kind = schema.get("type") if isinstance(schema, dict) else None
        if kind == "string":
            return value
        if kind in ("integer", "number", "boolean", "object", "array", "null") or kind is None:
            try:
                return json.loads(value)
            except ValueError:
                pass
            if kind == "boolean" and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
        return value

    def _parse_call(self, body: str) -> tuple[str, str] | None:
        m = _FUNC_RE.search(body)
        if not m:
            return None
        name, inner = m.group(1), m.group(2)
        props = self._param_types(name)
        args = {p: self._convert(v, props.get(p)) for p, v in _PARAM_RE.findall(inner)}
        return name, json.dumps(args, ensure_ascii=False)

    def extract_tool_calls(self, model_output, request) -> ExtractedToolCallInformation:
        if TOOL_START not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        calls = [c for c in map(self._parse_call, _CALL_RE.findall(model_output)) if c]
        if not calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        content = model_output[: model_output.index(TOOL_START)].strip() or None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[ToolCall(function=FunctionCall(name=n, arguments=a)) for n, a in calls],
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        self._buf += delta_text

        content = None
        start = self._buf.find(TOOL_START)
        if start < 0:
            safe_end = len(self._buf) - _held_back(self._buf, TOOL_START)
        else:
            safe_end = start
        if safe_end > self._content_sent:
            content = self._buf[self._content_sent : safe_end]
            self._content_sent = safe_end

        deltas = []
        for body in _CALL_RE.findall(self._buf)[self._calls_sent :]:
            self._calls_sent += 1
            parsed = self._parse_call(body)
            if not parsed:
                continue
            name, args = parsed
            index = len(self.prev_tool_call_arr)
            self.prev_tool_call_arr.append({"name": name, "arguments": json.loads(args)})
            self.streamed_args_for_tool.append(args)
            self.current_tool_id = index
            deltas.append(DeltaToolCall(
                index=index, id=ToolCall(function=FunctionCall(name=name, arguments=args)).id,
                type="function", function=DeltaFunctionCall(name=name, arguments=args),
            ))

        if content is None and not deltas:
            return None
        return DeltaMessage(content=content, tool_calls=deltas)


class ThinkTagsReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        ret = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids,
        )
        # The base class treats output as content when <think> was in the prompt rather than the output.
        if (ret is not None and self.start_token_id not in previous_token_ids
                and self.start_token_id not in delta_token_ids):
            if self.end_token_id in delta_token_ids:
                end = delta_text.find(self.end_token)
                rest = delta_text[end + len(self.end_token):]
                return DeltaMessage(reasoning=delta_text[:end], content=rest or None)
            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            return DeltaMessage(reasoning=delta_text)
        return ret


ToolParserManager.register_module("xml_tools", force=True, module=XmlToolParser)
ReasoningParserManager.register_module("think_tags", force=True, module=ThinkTagsReasoningParser)
