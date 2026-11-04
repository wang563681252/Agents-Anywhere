"""Adapt Hermes' ACP session controls to the shared runtime."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

from acp.connection import Connection, StreamDirection, StreamEvent

from ...domain.definitions import AcpError
from ...infrastructure.acp.client import AgentClient
from ...infrastructure.acp.v1 import wire

MODEL_OPTION = "hermes-model"
MODE_OPTION = "hermes-mode"
_REPLAY_UPDATES = {
    "agent_message_chunk",
    "agent_thought_chunk",
    "user_message_chunk",
    "tool_call",
    "tool_call_update",
}


class HermesClient(AgentClient):
    def __init__(self, on_update, on_interaction):
        self._on_update = on_update
        self._requests: dict[int | str, tuple[str, str | None]] = {}
        self._responses: dict[tuple[str, str], Any] = {}
        self._legacy: dict[str, dict[str, Any]] = {}
        self._native_options: dict[str, list[dict[str, Any]]] = {}
        self._replaying: set[str] = set()
        self._active: dict[str, tuple[str, str]] = {}
        self._observed: set[tuple[str, str]] = set()

        async def update(session_id, value):
            kind = value.get("sessionUpdate")
            if session_id in self._replaying and kind in _REPLAY_UPDATES:
                return
            if kind in {
                "agent_message_chunk",
                "agent_thought_chunk",
                "tool_call",
                "tool_call_update",
            }:
                await self._observe_execution(session_id)
            if kind == "current_mode_update" and session_id in self._legacy:
                mode = self._legacy[session_id].get("modes")
                if isinstance(mode, dict):
                    mode["currentModeId"] = value.get("currentModeId")
                    await self._on_update(
                        session_id,
                        {"sessionUpdate": "config_option_update", "configOptions": self._options(session_id)},
                    )
            elif kind == "config_option_update" and session_id in self._legacy:
                self._native_options[session_id] = value.get("configOptions", [])
                value = {**value, "configOptions": self._options(session_id)}
            await self._on_update(session_id, value)

        async def interaction(session_id, kind, value):
            if session_id:
                await self._observe_execution(session_id)
            return await on_interaction(session_id, kind, value)

        super().__init__(update, interaction)

    def _transport(self) -> Connection:
        # Hermes still returns ACP's legacy models and implements session/set_model.
        # The pinned SDK drops that response field and no longer exposes the method:
        # https://github.com/NousResearch/hermes-agent/blob/main/acp_adapter/server.py
        # https://github.com/agentclientprotocol/python-sdk/blob/0.12.1/src/acp/client/connection.py
        transport = getattr(self.connection, "_conn", None)
        if not isinstance(transport, Connection):
            raise AcpError("hermes_sdk_incompatible", "当前 ACP SDK 无法适配 Hermes 模型目录。")
        return transport

    def _observe_wire(self, event: StreamEvent) -> None:
        message = event.message
        identifier = message.get("id")
        if not isinstance(identifier, (int, str)):
            return
        if event.direction == StreamDirection.OUTGOING:
            method = message.get("method")
            if method in {"session/new", "session/load", "session/set_mode"}:
                params = message.get("params")
                session_id = params.get("sessionId") if isinstance(params, dict) else None
                self._requests[identifier] = (method, session_id)
        elif event.direction == StreamDirection.INCOMING:
            request = self._requests.pop(identifier, None)
            if not request or "result" not in message:
                return
            method, session_id = request
            result = message["result"]
            if method == "session/new" and isinstance(result, dict):
                session_id = result.get("sessionId")
            if isinstance(session_id, str) and session_id:
                self._responses[(method, session_id)] = (
                    {key: result[key] for key in ("models", "modes", "configOptions") if key in result}
                    if isinstance(result, dict) else result
                )

    async def start(self, launch, cwd):
        result = await super().start(launch, cwd)
        try:
            self._transport().add_observer(self._observe_wire)
            if self.capabilities.get("loadSession") is True:
                tracker = getattr(self.connection, "_session_updates", None)
                if not callable(getattr(tracker, "wait", None)):
                    raise AcpError(
                        "hermes_sdk_incompatible", "当前 ACP SDK 无法安全等待 Hermes 历史回放。"
                    )
                self.capabilities.setdefault("sessionCapabilities", {})["resume"] = {}
            return result
        except Exception:
            await self.close()
            raise

    @staticmethod
    def _legacy_option(
        state: Any, *, option_id: str, title: str, category: str, current_key: str,
        choices_key: str, value_key: str,
    ) -> dict[str, Any] | None:
        if state is None:
            return None
        if not isinstance(state, dict) or not isinstance(state.get(choices_key), list):
            raise AcpError("hermes_invalid_catalog", "Hermes 返回的会话配置无效。")
        current = state.get(current_key)
        choices = []
        seen = set()
        for item in state[choices_key]:
            if not isinstance(item, dict):
                raise AcpError("hermes_invalid_catalog", "Hermes 返回的会话配置无效。")
            value, name = item.get(value_key), item.get("name")
            if not isinstance(value, str) or not value or value in seen or not isinstance(name, str) or not name:
                raise AcpError("hermes_invalid_catalog", "Hermes 返回的会话配置无效。")
            seen.add(value)
            choice = {"value": value, "name": name}
            if isinstance(item.get("description"), str) and item["description"]:
                choice["description"] = item["description"]
            choices.append(choice)
        if not isinstance(current, str) or current not in seen:
            raise AcpError("hermes_invalid_catalog", "Hermes 当前选择不在其会话配置中。")
        choices.sort(key=lambda choice: choice["value"] != current)
        return {
            "id": option_id,
            "name": title,
            "category": category,
            "type": "select",
            "currentValue": current,
            "options": choices,
        }

    def _options(self, session_id: str) -> list[dict[str, Any]]:
        native = deepcopy(self._native_options.get(session_id, []))
        categories = {
            option.get("_meta", {}).get("aa", {}).get("selectionCategory")
            or option.get("category")
            for option in native
        }
        legacy = self._legacy[session_id]
        candidates = [
            self._legacy_option(
                legacy.get("models"),
                option_id=MODEL_OPTION,
                title="模型",
                category="model",
                current_key="currentModelId",
                choices_key="availableModels",
                value_key="modelId",
            ),
            self._legacy_option(
                legacy.get("modes"),
                option_id=MODE_OPTION,
                title="编辑审批",
                category="permission",
                current_key="currentModeId",
                choices_key="availableModes",
                value_key="id",
            ),
        ]
        existing_ids = {option.get("configId", option.get("id")) for option in native}
        for option in candidates:
            if option is None or option["category"] in categories:
                continue
            if option["id"] in existing_ids:
                raise AcpError("hermes_invalid_catalog", "Hermes 会话配置身份冲突。")
            native.append(option)
        return native

    def _save_session_options(self, session_id: str, raw: dict, result: dict) -> dict:
        self._legacy[session_id] = {
            "models": deepcopy(raw.get("models")),
            "modes": deepcopy(raw.get("modes", result.get("modes"))),
        }
        self._native_options[session_id] = deepcopy(result.get("configOptions", []))
        return {**result, "configOptions": self._options(session_id)}

    async def open_session(self, cwd, **metadata):
        result = await super().open_session(cwd, **metadata)
        session_id = result["sessionId"]
        raw = self._responses.pop(("session/new", session_id), None)
        if not isinstance(raw, dict):
            raise AcpError("hermes_session_unconfirmed", "Hermes 未返回可验证的会话信息。")
        return self._save_session_options(session_id, raw, result)

    async def resume(self, session_id, cwd, **metadata):
        if self.capabilities.get("loadSession") is not True:
            raise AcpError("hermes_resume_unsupported", "当前 Hermes 不支持恢复原会话。")
        self._replaying.add(session_id)
        try:
            response = await self.connection.load_session(
                session_id=session_id, cwd=cwd, mcp_servers=[], **metadata
            )
            # The pinned SDK does not wait for session/load notifications itself.
            await asyncio.sleep(0)
            await self.connection._session_updates.wait(session_id)
            raw = self._responses.pop(("session/load", session_id), None)
            # Hermes' ACP server can encode a missing session as an empty result.
            if not isinstance(raw, dict) or not (
                isinstance(raw.get("modes"), dict) or raw.get("configOptions")
            ):
                raise AcpError("hermes_session_missing", "Hermes 原会话不存在，未创建替代会话。")
            result = wire(response)
            if not isinstance(result, dict):
                raise AcpError("hermes_session_unconfirmed", "Hermes 未确认原会话恢复。")
            return self._save_session_options(session_id, raw, result)
        finally:
            self._replaying.discard(session_id)

    async def configure(self, session_id, key, value):
        options = self._options(session_id)
        option = next((item for item in options if item.get("configId", item.get("id")) == key), None)
        if option is None:
            raise AcpError("configuration_changed", "Hermes 会话配置已变化。")
        if key in {MODEL_OPTION, MODE_OPTION}:
            choices = {item["value"] for item in option["options"]}
            if not isinstance(value, str) or value not in choices:
                raise AcpError("invalid_configuration_value", "Hermes 会话配置值无效。")
            if value != option["currentValue"]:
                if key == MODEL_OPTION:
                    result = await self._transport().send_request(
                        "session/set_model", {"sessionId": session_id, "modelId": value}
                    )
                    if not isinstance(result, dict):
                        raise AcpError("hermes_model_unconfirmed", "Hermes 未确认模型切换。")
                    self._legacy[session_id]["models"]["currentModelId"] = value
                else:
                    await self.connection.set_session_mode(session_id=session_id, mode_id=value)
                    result = self._responses.pop(("session/set_mode", session_id), None)
                    if not isinstance(result, dict):
                        raise AcpError("hermes_mode_unconfirmed", "Hermes 未确认编辑审批模式。")
                    self._legacy[session_id]["modes"]["currentModeId"] = value
        else:
            result = await super().configure(session_id, key, value)
            self._native_options[session_id] = result.get("configOptions", [])
        return {"configOptions": self._options(session_id)}

    async def _observe_execution(self, session_id: str) -> None:
        active = self._active.get(session_id)
        if not active:
            return
        operation_id, text = active
        identity = (session_id, operation_id)
        if identity in self._observed:
            return
        self._observed.add(identity)
        await self._on_update(
            session_id,
            {
                "sessionUpdate": "user_message_chunk",
                "messageId": operation_id,
                "aaProjectionVerified": True,
                "content": {"type": "text", "text": text},
                "_meta": {"aa": {"clientMessageId": operation_id, "boundary": True}},
            },
        )
        await self._on_update(
            session_id, {"sessionUpdate": "aa_execution_observed", "clientMessageId": operation_id}
        )

    async def prompt(self, session_id, text, operation_id, blocks=None):
        self._active[session_id] = (operation_id, text)
        try:
            result = await super().prompt(session_id, text, operation_id, blocks)
            await self._observe_execution(session_id)
            return result
        finally:
            self._active.pop(session_id, None)
            self._observed.discard((session_id, operation_id))
