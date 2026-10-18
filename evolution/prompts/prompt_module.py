"""
HEOS System Prompt DSPy Module (Fixed & Type-Safe)
Maps system prompt sections as part of the signature instructions so GEPA can mutate them.
SRE Hardening: Persists evolved natural language guidance to JSON to avoid source-code corruption.
"""
import os
import json
import dspy
from pathlib import Path

from evolution.core.config import get_hermes_agent_path

class PromptSectionSignature(dspy.Signature):
    """Apply SRE behavioral guidelines to respond directly, accurately, and concisely following the system prompt section."""
    task_input = dspy.InputField(desc="The user task or raw system prompt section text to process.")
    agent_output = dspy.OutputField(desc="The structured, policy-compliant agent response or hardened prompt section.")


def get_evolution_repo_root() -> Path:
    """Return the root directory of the hermes-agent-self-evolution repository."""
    return Path(__file__).resolve().parents[2]


class PromptSectionModule(dspy.Module):
    def __init__(self, initial_guidance: str, json_path: Path | None = None):
        super().__init__()
        self.predictor = dspy.ChainOfThought(PromptSectionSignature)

        repo_root = get_evolution_repo_root()
        self.json_path = json_path or (repo_root / "output" / "evolved_system_prompts.json")

        if self.json_path.exists():
            try:
                self.guidance = json.loads(self.json_path.read_text())["guidance"]
            except Exception:
                self.guidance = initial_guidance
        else:
            self.guidance = initial_guidance

        self.predictor.predict.signature = self.predictor.predict.signature.with_instructions(self.guidance)  # type: ignore

    def update_guidance(self, new_guidance: str, section_name: str = "DEFAULT_AGENT_IDENTITY"):
        """SRE Hardened: Update signature instructions and persist to disk (No silent no-ops!)"""
        self.guidance = new_guidance
        sig = self.predictor.predict.signature

        if not hasattr(sig, "with_instructions"):
            raise RuntimeError("[SRE FAIL-CLOSED] Active DSPy Signature class lacks .with_instructions API!")

        new_sig = sig.with_instructions(new_guidance)  # type: ignore

        if new_sig.instructions != new_guidance:
            raise ValueError("[SRE FAIL-CLOSED] Failed to update DSPy signature instructions! Mutation was ignored.")

        self.predictor.predict.signature = new_sig

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if self.json_path.exists():
            try:
                existing = json.loads(self.json_path.read_text())
            except Exception:
                existing = {}
        existing["guidance"] = new_guidance
        sections = existing.get("sections", {})
        sections[section_name] = new_guidance
        existing["sections"] = sections
        self.json_path.write_text(json.dumps(existing, indent=2))

    def forward(self, task_input: str = "", raw_prompt_section: str = "", **kwargs):
        effective_input = task_input or raw_prompt_section or kwargs.get("input", "")
        res = self.predictor(task_input=effective_input)
        out_text = getattr(res, "agent_output", "") or ""
        # Expose both `output` (used by skill_fitness_metric) and `agent_output` (used by legacy callers)
        return dspy.Prediction(output=out_text, agent_output=out_text)
