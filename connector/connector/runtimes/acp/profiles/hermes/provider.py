"""Use the device's existing Hermes installation and configuration."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from connector.runtime_protocol import (
    RuntimeConfig,
    RuntimeConfigSchema,
    RuntimeInvalidRequestError,
    RuntimeProvider,
    RuntimeTypeDescriptor,
    RuntimeUnavailableError,
)

from ...domain.definitions import AgentProfile, LaunchDefinition
from .client import HermesClient
from .runtime import HermesRuntime

HERMES_PROFILE = AgentProfile(id="hermes", title="Hermes")


def _arguments(executable: Path) -> tuple[str, ...]:
    name = executable.name.lower()
    if name.endswith((".cmd", ".bat", ".ps1")):
        raise RuntimeInvalidRequestError(
            "请选择可直接执行的 Hermes 或 hermes-acp 入口，不支持脚本启动器。"
        )
    return () if name in {"hermes-acp", "hermes-acp.exe"} else ("acp",)


def resolve_launch(values: Mapping[str, Any]) -> LaunchDefinition:
    configured = values.get("executablePath")
    if configured:
        executable = Path(os.path.expandvars(configured)).expanduser()
        if not executable.is_absolute():
            raise RuntimeInvalidRequestError("Hermes 入口必须是绝对路径。")
    else:
        names = ("hermes.exe", "hermes-acp.exe") if os.name == "nt" else (
            "hermes",
            "hermes-acp",
        )
        found = next((path for name in names if (path := shutil.which(name))), None)
        if not found:
            home = Path.home()
            candidates = (
                [Path(os.environ["LOCALAPPDATA"]) / "hermes" / "bin" / "hermes.exe"]
                if os.name == "nt" and os.environ.get("LOCALAPPDATA")
                else [home / ".local" / "bin" / "hermes", home / ".local" / "bin" / "hermes-acp"]
            )
            found = next((str(path) for path in candidates if path.is_file()), None)
        if not found:
            raise RuntimeUnavailableError("未找到 Hermes，请先在工作设备安装 Hermes 或指定其入口。")
        executable = Path(found)
    if not executable.is_file() or (os.name != "nt" and not os.access(executable, os.X_OK)):
        raise RuntimeUnavailableError("Hermes 入口不存在或不可执行，请检查工作设备上的路径。")
    executable = executable.resolve()
    return LaunchDefinition(str(executable), _arguments(executable), "hermes-acp")


def check_launch(launch: LaunchDefinition) -> None:
    try:
        result = subprocess.run(
            [launch.executable, *launch.arguments, "--check"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeUnavailableError(
            "Hermes ACP 检查无法完成，请在工作设备运行 hermes acp --check。"
        ) from error
    if result.returncode:
        raise RuntimeUnavailableError(
            "Hermes ACP 不可用，请在工作设备运行 hermes acp --check 并安装 ACP 依赖。"
        )


class HermesProvider(RuntimeProvider):
    runtime_type = "hermes"
    display_name = "Hermes"
    implementation_type = "acp-stdio"
    instance_policy = "single"

    @property
    def description(self) -> str:
        return "Hermes Agent using the device's existing ACP installation"

    async def get_config_schema(self) -> RuntimeConfigSchema:
        return RuntimeConfigSchema(
            runtime=self.runtime_type,
            revision=1,
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "executablePath": {
                        "type": "string",
                        "title": "Hermes 或 hermes-acp 的绝对路径（可选）",
                        "minLength": 1,
                        "maxLength": 4096,
                    },
                },
            },
            ui_schema={"order": ["executablePath"], "executablePath": {"component": "path"}},
            defaults={},
        )

    async def discover(self) -> RuntimeTypeDescriptor:
        return RuntimeTypeDescriptor(
            runtime_type=self.runtime_type,
            display_name=self.display_name,
            description=self.description,
            available=True,
            implementation_type=self.implementation_type,
            instance_policy=self.instance_policy,
            max_instances=1,
            config_schema=await self.get_config_schema(),
            capabilities={"workspacePreparation": True, "asyncCreation": True},
            metadata={"localInstallation": True, "nativeConfiguration": True},
        )

    async def validate_config(self, values: Mapping[str, Any]) -> RuntimeConfig:
        schema = await self.get_config_schema()
        errors = list(Draft202012Validator(schema.schema).iter_errors(dict(values)))
        if errors:
            raise RuntimeInvalidRequestError(f"Hermes 配置无效：{errors[0].message}")
        if values.get("executablePath"):
            path = Path(os.path.expandvars(values["executablePath"])).expanduser()
            if not path.is_absolute():
                raise RuntimeInvalidRequestError("Hermes 入口必须是绝对路径。")
        return RuntimeConfig(
            runtime=self.runtime_type,
            revision=schema.revision,
            values=dict(values),
            schema=schema.schema,
            ui_schema=schema.ui_schema,
        )

    async def create_runtime(self, config: RuntimeConfig, host) -> HermesRuntime:
        config = replace(config, runtime_id=getattr(host, "runtime_id", None) or config.runtime_id)
        launch = resolve_launch(config.values)
        await asyncio.to_thread(check_launch, launch)
        return HermesRuntime(config, host, HERMES_PROFILE, launch, client_factory=HermesClient)


def hermes_acp_provider() -> HermesProvider:
    return HermesProvider()
