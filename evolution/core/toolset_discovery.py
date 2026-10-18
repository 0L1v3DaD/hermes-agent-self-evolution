"""Discover currently-active Hermes toolsets — used to inform the optimizer.

The GEPA optimizer was observed (2026-05-10) to hallucinate capability denial
("you do not have access to filesystem / browser / delegate_task") because it
generalized from sub-agent / sandbox examples in the dataset. This module gives
the optimizer ground truth about which toolsets are actually wired so its
reflection step has no excuse to invent limitations.

Source of truth: ``TOOLSETS`` dict in ``hermes-agent/toolsets.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _find_hermes_toolsets_module() -> Optional[Path]:
    """Locate the toolsets.py owned by the local hermes-agent install.

    Search order:
      1. $HERMES_AGENT_REPO env var (set by the evolve_skill.py runner)
      2. ~/.hermes/hermes-agent/toolsets.py (the conventional install path)
      3. None
    """
    candidates: list[Path] = []
    env_path = os.environ.get("HERMES_AGENT_REPO")
    if env_path:
        candidates.append(Path(env_path) / "toolsets.py")
    candidates.append(Path.home() / ".hermes" / "hermes-agent" / "toolsets.py")
    for p in candidates:
        if p.is_file():
            return p
    return None


def _discover_mcp_toolsets() -> list[str]:
    """Discover enabled MCP servers from ~/.hermes/config.yaml (e.g., mcp:rea)."""
    cfg_path = Path.home() / ".hermes" / "config.yaml"
    if not cfg_path.is_file():
        return []
    try:
        import yaml
        data = yaml.safe_load(cfg_path.read_text()) or {}
        mcp_servers = data.get("mcp_servers") or {}
        if not isinstance(mcp_servers, dict):
            return []
        enabled = []
        for name, spec in mcp_servers.items():
            if isinstance(spec, dict) and spec.get("enabled", True) is False:
                continue
            enabled.append(f"mcp:{name}")
        return sorted(enabled)
    except Exception:
        return []


def discover_available_toolsets() -> list[str]:
    """Return the list of currently-defined agent-facing toolset names from toolsets.py
    plus active MCP servers from ~/.hermes/config.yaml (such as mcp:rea).

    Filters out ``hermes-*`` platform/adapter entries (gateway integrations like
    hermes-telegram, hermes-discord, hermes-webhook). Those exist in the TOOLSETS
    dict but aren't tools the agent invokes — keeping them out of the optimizer's
    context window keeps the signal/noise high.

    Returns an empty list (and logs nothing — optimizer should still work) if
    toolsets.py is not findable, so this never blocks an evolve run.
    """
    path = _find_hermes_toolsets_module()
    if path is None:
        return []

    try:
        import ast

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TOOLSETS":
                        if isinstance(node.value, ast.Dict):
                            all_keys = [
                                k.value
                                for k in node.value.keys
                                if isinstance(k, ast.Constant) and isinstance(k.value, str)
                            ]
                            agent_facing = [k for k in all_keys if not k.startswith("hermes-")]
                            mcp_keys = _discover_mcp_toolsets()
                            return sorted(set(agent_facing + mcp_keys))
        return []
    except Exception:
        return []


def format_toolsets_for_prompt() -> str:
    """Return a one-line, agent-readable summary of available toolsets.

    Designed to be injected into the SkillModule signature as an InputField
    so GEPA's reflection step has explicit ground truth about capabilities.
    Format kept short to avoid bloating every prediction's input cost.
    """
    toolsets = discover_available_toolsets()
    if not toolsets:
        return "(toolset discovery unavailable — assume standard agent capabilities)"
    return (
        "The following toolsets and MCP servers are AVAILABLE to the agent at runtime — "
        "DO NOT write skill text claiming the agent lacks any of these capabilities: "
        + ", ".join(toolsets)
    )
