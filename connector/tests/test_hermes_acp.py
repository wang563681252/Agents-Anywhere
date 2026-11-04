from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from connector.runtime_protocol import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeIdentity,
    RuntimeInstance,
    RuntimeInstanceSpec,
    RuntimeInvalidRequestError,
    RuntimePermissionCatalog,
    RuntimePermissionItem,
    RuntimeUnavailableError,
)
from connector.runtimes.acp.application.sessions import SessionService
from connector.runtimes.acp.domain.definitions import AcpError, LaunchDefinition
from connector.runtimes.acp.infrastructure.persistence import Store
from connector.runtimes.acp.infrastructure.projection import Projection
from connector.runtimes.acp.profiles.hermes.client import (
    MODE_OPTION,
    MODEL_OPTION,
    HermesClient,
)
from connector.runtimes.acp.profiles.hermes.provider import (
    HERMES_PROFILE,
    HermesProvider,
    check_launch,
    resolve_launch,
)
from connector.runtimes.acp.profiles.hermes.runtime import HermesRuntime
from connector.server.runtime_rpc import RuntimeRpcHandler
from connector.server.runtime_rpc_payloads import runtime_type_descriptor_payload


def test_hermes_provider_discovery_and_configuration(tmp_path):
    async def verify():
        provider = HermesProvider()
        descriptor = await provider.discover()
        assert descriptor.runtime_type == "hermes"
        assert descriptor.display_name == "Hermes"
        assert descriptor.implementation_type == "acp-stdio"
        assert descriptor.instance_policy == "single"
        schema_path = (
            Path(__file__).resolve().parents[2] / "contracts" / "runtime-control" / "2.0"
            / "schemas" / "runtime-type-descriptor.schema.json"
        )
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
            runtime_type_descriptor_payload(descriptor)
        )
        assert (await provider.validate_config({})).runtime == "hermes"
        with pytest.raises(RuntimeInvalidRequestError):
            await provider.validate_config({"unexpected": "value"})
        with pytest.raises(RuntimeInvalidRequestError):
            await provider.validate_config({"executablePath": "relative/hermes"})
        executable = tmp_path / ("hermes-acp.exe" if os.name == "nt" else "hermes-acp")
        executable.touch()
        executable.chmod(0o700)
        config = await provider.validate_config({"executablePath": str(executable)})
        launch = resolve_launch(config.values)
        assert launch.executable == str(executable.resolve())
        assert launch.arguments == ()

    asyncio.run(verify())


def test_hermes_launcher_missing_and_failed_check_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("connector.runtimes.acp.profiles.hermes.provider.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    with pytest.raises(RuntimeUnavailableError, match="未找到 Hermes"):
        resolve_launch({})
    executable = tmp_path / ("hermes.exe" if os.name == "nt" else "hermes")
    executable.touch()
    executable.chmod(0o700)
    launch = resolve_launch({"executablePath": str(executable)})
    assert launch.arguments == ("acp",)
    monkeypatch.setattr(
        "connector.runtimes.acp.profiles.hermes.provider.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, b"secret", b"secret"),
    )
    with pytest.raises(RuntimeUnavailableError) as error:
        check_launch(launch)
    assert "secret" not in str(error.value)
    assert "hermes acp --check" in str(error.value)


def test_hermes_permission_catalog_is_scoped_through_runtime_rpc(tmp_path):
    async def verify():
        host = SimpleNamespace(
            session_namespace="fixture",
            runtime_kv=SimpleNamespace(path=tmp_path / "kv.json"),
            runtime_id="hermes",
        )
        native = HermesRuntime(
            RuntimeConfig(runtime="hermes", revision=1, runtime_id="hermes"),
            host,
            HERMES_PROFILE,
            LaunchDefinition(sys.executable, (), "fixture"),
        )
        try:
            for session_id, mode in (("session-a", "default"), ("session-b", "accept_edits")):
                native.app.store.put("binding", session_id, {
                    "session_id": session_id,
                    "agent_session_id": "native-" + session_id,
                    "workspace": {"registered_path": str(tmp_path)},
                })
                native.app.configurations.observe(session_id, [{
                    "id": MODE_OPTION,
                    "category": "permission",
                    "type": "select",
                    "currentValue": mode,
                    "options": [{"value": mode, "name": mode}],
                }])
            async def no_native_inventory():
                raise AssertionError("Hermes native history must not be discovered")
            native.app.inventory_client = no_native_inventory
            sessions = await native.list_sessions()
            assert {item.session_id for item in sessions} == {"session-a", "session-b"}
            wrapped = RuntimeInstance(
                RuntimeInstanceSpec(runtime_id="hermes", runtime_type="hermes", name="Hermes"),
                native,
            )
            supervisor = SimpleNamespace(resolve_runtime=lambda *_args: wrapped)
            handler = RuntimeRpcHandler(supervisor, host)
            for session_id, expected in (("session-a", "default"), ("session-b", "accept_edits")):
                result = await handler.dispatch("runtime.permissionCatalog", {
                    "runtime": "hermes", "runtimeId": "hermes", "sessionId": session_id,
                })
                assert result["catalog"]["runtimeId"] == "hermes"
                assert [item["selectionId"] for item in result["catalog"]["permissions"]] == [expected]
                schema_path = (
                    Path(__file__).resolve().parents[2] / "contracts" / "protocol" / "1.1"
                    / "schemas" / "permission-catalog.schema.json"
                )
                Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
                    result["catalog"]
                )
            unscoped = await handler.dispatch("runtime.permissionCatalog", {
                "runtime": "hermes", "runtimeId": "hermes",
            })
            assert unscoped["catalog"]["permissions"] == []
        finally:
            native.app.store.close()

    asyncio.run(verify())


def test_scoped_permission_catalog_preserves_legacy_runtime_behavior():
    class LegacyRuntime(AgentRuntime):
        @property
        def identity(self):
            return RuntimeIdentity(runtime="codex", runtime_version="fixture")

        async def list_permission_catalog(self, query=None, limit=100):
            assert query == "read" and limit == 5
            return RuntimePermissionCatalog(
                runtime="codex",
                revision=1,
                permissions=(RuntimePermissionItem(id="read", title="Read", selection_id="read"),),
            )

    async def verify():
        wrapped = RuntimeInstance(
            RuntimeInstanceSpec(runtime_id="codex", runtime_type="codex", name="Codex"),
            LegacyRuntime(),
        )
        handler = RuntimeRpcHandler(SimpleNamespace(resolve_runtime=lambda *_args: wrapped), None)
        result = await handler.dispatch("runtime.permissionCatalog", {
            "runtime": "codex", "runtimeId": "codex", "sessionId": "session",
            "query": "read", "limit": 5,
        })
        assert result["catalog"]["runtimeId"] == "codex"
        assert result["catalog"]["permissions"][0]["selectionId"] == "read"

    asyncio.run(verify())


def test_hermes_process_crash_closes_pending_approval_and_allows_recovery(tmp_path):
    async def verify():
        async def until(predicate):
            async with asyncio.timeout(10):
                while not predicate():
                    await asyncio.sleep(0.02)

        peer = Path(__file__).with_name("hermes_acp_probe.py")
        store = Store(tmp_path / "state.sqlite")
        app = SessionService(
            store,
            HERMES_PROFILE,
            LaunchDefinition(sys.executable, (str(peer), str(tmp_path / "write")), "fixture"),
            HermesClient,
            Projection,
            tmp_path,
        )
        try:
            app.prepare({"preparationId": "prep", "clientPreparationId": "client", "cwd": str(tmp_path)})
            await until(lambda: app.preparation_view("prep")["phase"] != "opening")
            prepared = app.preparation_view("prep")
            assert prepared["phase"] == "ready", prepared
            assert app.reserve({
                "preparationId": "prep", "createOperationId": "create",
                "platformSessionId": "session", "expectedRevision": prepared["revision"],
                "catalogRevision": prepared["catalogRevision"],
            }) == {"reserved": True}
            app.begin({
                "preparationId": "prep", "createOperationId": "create",
                "sessionId": "session", "clientMessageId": "first", "content": "synthetic",
            })
            await until(lambda: any(
                item["status"] == "open" for item in app.interactions.views("create")
            ))
            await app.sessions["session"]["client"].process.close()
            await until(lambda: app.state("session")["status"] == "blocked" and not app.sessions)
            assert all(item["status"] != "open" for item in app.interactions.views("create"))
            await app.recover("session")
            assert app.state("session")["status"] == "idle"
        finally:
            await app.stop()
            store.close()

    asyncio.run(verify())


@pytest.mark.parametrize("decision,expected_write", [("allow_once", True), ("deny", False)])
def test_hermes_real_acp_transport_catalog_approval_and_recovery(tmp_path, decision, expected_write):
    async def verify():
        updates = []
        requests = []

        async def on_update(session_id, update):
            updates.append((session_id, update))

        async def on_interaction(session_id, kind, request):
            requests.append((session_id, kind, request))
            return {"action": "accept", "optionId": decision}

        client = HermesClient(on_update, on_interaction)
        marker = tmp_path / "synthetic-write"
        peer = Path(__file__).with_name("hermes_acp_probe.py")
        launch = LaunchDefinition(sys.executable, (str(peer), str(marker)), "fixture")
        try:
            await client.start(launch, str(tmp_path))
            session = await client.open_session(str(tmp_path))
            session_id = session["sessionId"]
            options = {item["id"]: item for item in session["configOptions"]}
            assert options[MODEL_OPTION]["currentValue"] == "fixture:first"
            assert options[MODE_OPTION]["currentValue"] == "default"
            assert options[MODEL_OPTION]["options"][0]["value"] == "fixture:first"
            assert options[MODE_OPTION]["options"][0]["value"] == "default"
            result = await client.configure(session_id, MODEL_OPTION, "fixture:second")
            assert next(item for item in result["configOptions"] if item["id"] == MODEL_OPTION)["currentValue"] == "fixture:second"
            result = await client.configure(session_id, MODE_OPTION, "accept_edits")
            assert next(item for item in result["configOptions"] if item["id"] == MODE_OPTION)["currentValue"] == "accept_edits"
            with pytest.raises(AcpError, match="未确认模型切换"):
                await client.configure(session_id, MODEL_OPTION, "fixture:unconfirmed")
            with pytest.raises(AcpError, match="配置值无效"):
                await client.configure(session_id, MODE_OPTION, "not-a-mode")

            await client.prompt(session_id, "synthetic user", "message-1")
            kinds = [value["sessionUpdate"] for _, value in updates]
            assert kinds.count("user_message_chunk") == 1
            assert kinds.count("aa_execution_observed") == 1
            assert kinds.count("agent_message_chunk") == 1
            assert len(requests) == 1 and requests[0][1] == "permission"
            assert marker.exists() is expected_write

            before = len(updates)
            resumed = await client.resume(session_id, str(tmp_path))
            assert resumed["configOptions"]
            assert not any(
                item["sessionUpdate"] in {"user_message_chunk", "agent_message_chunk"}
                for _, item in updates[before:]
            )
            with pytest.raises(AcpError) as error:
                await client.resume("missing-session", str(tmp_path))
            assert error.value.code == "hermes_session_missing"
        finally:
            await client.close()

    asyncio.run(verify())
