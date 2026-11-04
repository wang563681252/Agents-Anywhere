from __future__ import annotations

from connector.runtime_protocol.models import (
    RuntimePermissionCatalog,
    RuntimePermissionItem,
    SessionMeta,
)

from ...runtime import AcpRuntime


class HermesRuntime(AcpRuntime):
    async def list_sessions(self, limit=100, cursor=None, force=False):
        rows = [
            SessionMeta(
                session_id=row["session_id"],
                external_session_id=row["agent_session_id"],
                runtime=self.config.runtime,
                cwd=row["workspace"]["registered_path"],
            )
            for row in self.app.store.values("binding")
        ]
        return tuple(rows[:limit])

    async def list_scoped_permission_catalog(
        self, session_id: str, query: str | None = None, limit: int = 100
    ) -> RuntimePermissionCatalog:
        self.app.store.require("binding", session_id)
        snapshot = self.app.configurations.snapshot(session_id)
        choices = [
            item
            for option in snapshot.options
            if option.get("category") == "permission" and option.get("type") == "select"
            for group in option.get("options", [])
            for item in group.get("options", [group])
        ]
        if query:
            needle = query.casefold()
            choices = [
                item for item in choices
                if needle in item["value"].casefold() or needle in item["name"].casefold()
            ]
        return RuntimePermissionCatalog(
            runtime=self.config.runtime,
            revision=snapshot.revision,
            permissions=tuple(
                RuntimePermissionItem(
                    id=item["value"],
                    title=item["name"],
                    selection_id=item["value"],
                    description=item.get("description"),
                )
                for item in choices[:limit]
            ),
        )
