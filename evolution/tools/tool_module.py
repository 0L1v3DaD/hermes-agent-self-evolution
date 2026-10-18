"""
HEOS Tool Selection DSPy Module (Fixed & Type-Safe)
Directly maps tool descriptions as part of the signature instructions.
SRE Hardening: Persists evolved natural language schemas to JSON to avoid source-code corruption.
"""
import os
import json
import dspy
from pathlib import Path

from evolution.core.config import get_hermes_agent_path

class TaskToTool(dspy.Signature):
    """Given a high-level user task instruction, select the single best tool from the whitelisted set (`file_tools`, `terminal_tool`, `mcp:rea`, `web_tools`, `memory_wiki`) and extract JSON parameters.
    - Use `file_tools` for reading, writing, or inspecting local source/config files (`{"path": "<filepath>"}`).
    - Use `terminal_tool` for running tests, git, docker, or shell commands (`{"command": "<cmd>"}`).
    - Use `mcp:rea` for reverse-engineering binaries, APKs, or packaged JS/Electron bundles (`{"target": "<path>"}`).
    """
    task_input = dspy.InputField(desc="The user instruction/task to execute.")
    selected_tool = dspy.OutputField(desc="Name of the selected tool (e.g., file_tools, terminal_tool, mcp:rea).")
    parameter_values = dspy.OutputField(desc="JSON-serialized dictionary of extracted parameter key-value pairs.")


def get_evolution_repo_root() -> Path:
    """Return the root directory of the hermes-agent-self-evolution repository."""
    return Path(__file__).resolve().parents[2]


class ToolSelectionModule(dspy.Module):
    def __init__(self, initial_descriptions: str, json_path: Path | None = None):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TaskToTool)

        # Persist evolved descriptions to the evolution repo's output/ directory
        repo_root = get_evolution_repo_root()
        self.json_path = json_path or (repo_root / "output" / "evolved_tool_descriptions.json")

        if self.json_path.exists():
            try:
                self.descriptions = json.loads(self.json_path.read_text())["descriptions"]
            except Exception:
                self.descriptions = initial_descriptions
        else:
            self.descriptions = initial_descriptions

        self.predictor.predict.signature = self.predictor.predict.signature.with_instructions(self.descriptions)  # type: ignore

    def update_descriptions(self, new_descriptions: str):
        """SRE Hardened: Update signature instructions and persist to disk (No silent no-ops!)"""
        self.descriptions = new_descriptions
        sig = self.predictor.predict.signature

        if not hasattr(sig, "with_instructions"):
            raise RuntimeError("[SRE FAIL-CLOSED] Active DSPy Signature class lacks .with_instructions API!")

        new_sig = sig.with_instructions(new_descriptions)  # type: ignore

        if new_sig.instructions != new_descriptions:
            raise ValueError("[SRE FAIL-CLOSED] Failed to update DSPy signature instructions! Mutation was ignored.")

        self.predictor.predict.signature = new_sig

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps({"descriptions": new_descriptions}, indent=2))

    def forward(self, task_input: str):
        return self.predictor(task_input=task_input)
