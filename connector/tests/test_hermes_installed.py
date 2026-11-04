"""Optional real-Hermes smoke test using only a temporary home and loopback model."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from connector.runtimes.acp.profiles.hermes.client import (
    MODE_OPTION,
    MODEL_OPTION,
)
from connector.runtimes.acp.profiles.hermes.provider import HermesProvider


@pytest.mark.skipif(
    not os.environ.get("AA_HERMES_ACP_TEST_EXECUTABLE"),
    reason="set AA_HERMES_ACP_TEST_EXECUTABLE to a local hermes-acp executable",
)
def test_installed_hermes_workflow_with_loopback_model(tmp_path, monkeypatch):
    executable = Path(os.environ["AA_HERMES_ACP_TEST_EXECUTABLE"])
    assert executable.is_absolute() and executable.is_file()
    for key in tuple(os.environ):
        if any(part in key.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")):
            monkeypatch.delenv(key, raising=False)

    requests = []
    advertised_tools = []

    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            pass

        def do_GET(self):
            body = json.dumps({
                "object": "list",
                "data": [
                    {"id": "fixture-model", "object": "model"},
                    {"id": "fixture-model-2", "object": "model"},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request.get("model")))
            messages = request.get("messages", [])
            tools = request.get("tools", [])
            if self.path.endswith("/chat/completions"):
                advertised_tools.append([
                    item.get("function", {}).get("name") for item in tools
                ])
            last_user = next(
                (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
                -1,
            )
            text = messages[last_user].get("content") if last_user >= 0 else None
            scenario = {
                "Synthetic denied edit": "denied.txt",
                "Synthetic allowed edit": "allowed.txt",
            }.get(text) if isinstance(text, str) else None
            tool_result = any(item.get("role") == "tool" for item in messages[last_user + 1:])
            tool_call = None
            if scenario and not tool_result:
                tool_call = {
                    "id": "fixture-edit",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": str(workspace / scenario), "content": "synthetic edit"}),
                    },
                }
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                first = (
                    {"role": "assistant", "tool_calls": [{"index": 0, **tool_call}]}
                    if tool_call else {"role": "assistant", "content": "synthetic model reply"}
                )
                for delta, reason in [(first, None), ({}, "tool_calls" if tool_call else "stop")]:
                    chunk = {
                        "id": "fixture-completion",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "fixture-model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                    }
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            message = (
                {"role": "assistant", "content": None, "tool_calls": [tool_call]}
                if tool_call else {"role": "assistant", "content": "synthetic model reply"}
            )
            body = json.dumps({
                "id": "fixture-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_call else "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    model = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=model.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: fixture-model\n"
        "  provider: custom:fixture\n"
        f"  base_url: http://127.0.0.1:{model.server_port}/v1\n"
        "providers:\n"
        "  fixture:\n"
        "    name: fixture\n"
        f"    base_url: http://127.0.0.1:{model.server_port}/v1\n"
        "    default_model: fixture-model\n"
        "    discover_models: false\n"
        "    models:\n"
        "      fixture-model: {}\n"
        "      fixture-model-2: {}\n",
        encoding="utf-8",
    )
    for key, value in {
        "HERMES_HOME": str(home),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "NO_PROXY": "127.0.0.1,localhost",
    }.items():
        monkeypatch.setenv(key, value)

    async def until(predicate):
        async with asyncio.timeout(90):
            while not predicate():
                await asyncio.sleep(0.05)

    async def verify():
        class Host:
            session_namespace = "hermes-smoke"
            runtime_id = "hermes"
            runtime_kv = SimpleNamespace(path=tmp_path / "runtime" / "kv.json")

            async def publish_runtime_notifications(self, _runtime, _batch, **_kwargs):
                return None

        provider = HermesProvider()
        config = await provider.validate_config({"executablePath": str(executable)})
        host = Host()
        runtime = await provider.create_runtime(config, host)
        await runtime.start()
        app = runtime.app
        store = app.store
        try:
            app.prepare({"preparationId": "prep", "clientPreparationId": "client", "cwd": str(workspace)})
            await until(lambda: app.preparation_view("prep")["phase"] != "opening")
            prepared = app.preparation_view("prep")
            assert prepared["phase"] == "ready", prepared
            schema_path = (
                Path(__file__).resolve().parents[2] / "contracts" / "protocol" / "1.1"
                / "schemas" / "model-catalog.schema.json"
            )
            Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
                prepared["catalog"]
            )
            options = {item["id"]: item for item in prepared["configuration"]["options"]}
            assert MODEL_OPTION in options and MODE_OPTION in options
            assert options[MODEL_OPTION]["options"][0]["value"] == options[MODEL_OPTION]["currentValue"]
            selected_model = next(
                item["value"] for item in options[MODEL_OPTION]["options"]
                if item["value"].endswith(":fixture-model-2")
            )
            selections = {
                "model": selected_model,
                "permission": options[MODE_OPTION]["currentValue"],
            }
            reservation = {
                "preparationId": "prep",
                "createOperationId": "create",
                "platformSessionId": "session",
                "expectedRevision": prepared["revision"],
                "catalogRevision": prepared["catalogRevision"],
                "selections": selections,
            }
            assert app.reserve(reservation) == {"reserved": True}
            app.begin({
                "preparationId": "prep",
                "createOperationId": "create",
                "sessionId": "session",
                "clientMessageId": "first",
                "content": "Synthetic integration prompt",
                "selections": selections,
            })
            await until(lambda: app.creation_view("create")["messageAccepted"] is not None)
            await until(lambda: app.state("session")["status"] == "idle")
            assert app.creation_view("create")["messageAccepted"] is True
            items = [item for item in store.values("projection") if item["sessionId"] == "session"]
            assert any(item["role"] == "user" and item["content"]["text"] == "Synthetic integration prompt" for item in items)
            assert any(item["role"] == "assistant" and "synthetic model reply" in item["content"]["text"] for item in items)
            assert ("/v1/chat/completions", "fixture-model-2") in requests
            assert any("write_file" in tools for tools in advertised_tools)
            catalog = await runtime.list_scoped_permission_catalog("session")
            assert {item.id for item in catalog.permissions} == {"default", "accept_edits", "dont_ask"}
            assert [item.session_id for item in await runtime.list_sessions()] == ["session"]

            async def edit(prompt, operation, choice):
                await app.submit("session", operation, prompt)
                await until(lambda: any(
                    item["status"] == "open" for item in app.interactions.views("session")
                ))
                request = next(
                    item for item in app.interactions.views("session") if item["status"] == "open"
                )
                app.interactions.respond("session", {
                    "interactionId": request["interactionId"],
                    "ownerGeneration": request["ownerGeneration"],
                    "requestRevision": request["requestRevision"],
                    "responseOperationId": operation + "-answer",
                    "actionId": choice,
                })
                await until(lambda: app.state("session")["status"] == "idle")

            await edit("Synthetic denied edit", "deny-edit", "deny")
            assert not (workspace / "denied.txt").exists()
            await edit("Synthetic allowed edit", "allow-edit", "allow_once")
            assert (workspace / "allowed.txt").read_text(encoding="utf-8") == "synthetic edit"
            native_id = app.state("session")["externalSessionId"]
            mode = await runtime.update_session_selections(
                "session", native_id, {"permission": "accept_edits"}
            )
            assert mode.ok
            mode = await runtime.update_session_selections(
                "session", native_id, {"permission": "default"}
            )
            assert mode.ok
            original_model = options[MODEL_OPTION]["currentValue"]
            model_change = await runtime.update_session_selections(
                "session", native_id, {"model": original_model}
            )
            assert model_change.ok
        finally:
            await runtime.stop()

        runtime = await provider.create_runtime(config, host)
        await runtime.start()
        app = runtime.app
        store = app.store
        try:
            before = len(store.values("projection"))
            await app.recover("session")
            assert len(store.values("projection")) == before
            assert app.state("session")["status"] == "idle"
            await app.submit("session", "second", "Another synthetic prompt")
            await until(lambda: app.state("session")["status"] == "idle")
            assert ("/v1/chat/completions", "fixture-model") in requests
        finally:
            await runtime.stop()

    try:
        asyncio.run(verify())
    finally:
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)
