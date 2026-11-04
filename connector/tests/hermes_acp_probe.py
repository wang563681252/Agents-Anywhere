"""A controlled Hermes-shaped ACP peer with no model or external network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
MARKER = Path(sys.argv[1])
SESSION = "hermes-fixture-session"
current_model = "fixture:first"
current_mode = "default"


def send(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def response(identifier, result):
    send({"jsonrpc": "2.0", "id": identifier, "result": result})


def update(kind, text):
    send({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": SESSION,
            "update": {"sessionUpdate": kind, "content": {"type": "text", "text": text}},
        },
    })


def state():
    return {
        "models": {
            "currentModelId": current_model,
            "availableModels": [
                {"modelId": "fixture:second", "name": "Second"},
                {"modelId": "fixture:first", "name": "First"},
                {"modelId": "fixture:unconfirmed", "name": "Unconfirmed"},
            ],
        },
        "modes": {
            "currentModeId": current_mode,
            "availableModes": [
                {"id": "accept_edits", "name": "Accept Edits"},
                {"id": "default", "name": "Default", "description": "Ask before edits"},
            ],
        },
    }


for line in sys.stdin.buffer:
    request = json.loads(line)
    method = request.get("method")
    identifier = request.get("id")
    params = request.get("params") or {}
    if method == "initialize":
        response(identifier, {
            "protocolVersion": 1,
            "agentCapabilities": {"loadSession": True, "sessionCapabilities": {"resume": {}}},
            "agentInfo": {"name": "hermes-fixture", "version": "1"},
        })
    elif method == "session/new":
        response(identifier, {"sessionId": SESSION, **state()})
    elif method == "session/load":
        if params["sessionId"] != SESSION:
            response(identifier, {})
        else:
            update("user_message_chunk", "replayed user")
            update("agent_message_chunk", "replayed assistant")
            response(identifier, state())
    elif method == "session/set_model":
        if params["modelId"] == "fixture:unconfirmed":
            response(identifier, None)
        else:
            current_model = params["modelId"]
            response(identifier, {})
    elif method == "session/set_mode":
        current_mode = params["modeId"]
        response(identifier, {})
    elif method == "session/prompt":
        send({
            "jsonrpc": "2.0",
            "id": "permission-fixture",
            "method": "session/request_permission",
            "params": {
                "sessionId": SESSION,
                "toolCall": {
                    "toolCallId": "tool-fixture",
                    "title": "Synthetic write",
                    "kind": "edit",
                    "status": "pending",
                },
                "options": [
                    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
                ],
            },
        })
        answer = json.loads(sys.stdin.buffer.readline())
        outcome = answer.get("result", {}).get("outcome", {})
        if outcome.get("outcome") == "selected" and outcome.get("optionId") == "allow_once":
            MARKER.write_text("approved", encoding="utf-8")
        update("agent_message_chunk", "synthetic reply")
        response(identifier, {"stopReason": "end_turn"})
