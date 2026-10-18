"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name across Hermes and shared CLI harness skill directories.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    Search order:
      1. <hermes_agent_path>/skills
      2. ~/.hermes/skills
      3. ~/.agents/skills
      4. ~/.gemini/config/skills
    """
    search_roots = [
        hermes_agent_path / "skills",
        Path.home() / ".hermes" / "skills",
        Path.home() / ".agents" / "skills",
        Path.home() / ".gemini" / "config" / "skills",
    ]
    existing_roots = [d for d in search_roots if d.exists()]
    if not existing_roots:
        return None

    # Pass 1: Direct directory name match: <root>/.../<skill_name>/SKILL.md
    for root in existing_roots:
        for skill_md in root.rglob("SKILL.md"):
            if skill_md.parent.name == skill_name:
                return skill_md

    # Pass 2: Fuzzy match on frontmatter `name:` field
    for root in existing_roots:
        for skill_md in root.rglob("SKILL.md"):
            try:
                content = skill_md.read_text()[:500]
                if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                    return skill_md
            except Exception:
                continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text (body) is the parameter that GEPA optimizes.
    On each forward pass, the module:
    1. Uses the skill text as instructions
    2. Processes the task input
    3. Returns the agent's response

    The signature also exposes ``available_toolsets`` as an InputField so the
    GEPA optimizer's reflection step has explicit ground truth about agent
    capabilities. This counters a documented failure mode (2026-05-10) where
    the optimizer hallucinated capability denial in evolved prompts because
    it generalized from a few sub-agent / sandbox examples in the dataset.
    """

    class TaskWithSkill(dspy.Signature):
        """Complete a task following the provided skill instructions.

        You are an AI agent following specific skill instructions to complete a task.
        Read the skill instructions carefully and follow the procedure described.

        IMPORTANT — capability ground truth: the ``available_toolsets`` field below
        lists every toolset wired into the agent's runtime. Do NOT include skill
        text that claims the agent lacks a capability covered by these toolsets.
        Skill text should attempt the relevant tool, not refuse on imagined limits.
        """
        skill_instructions: str = dspy.InputField(desc="The skill instructions to follow")
        task_input: str = dspy.InputField(desc="The task to complete")
        available_toolsets: str = dspy.InputField(
            desc="Comma-separated list of toolsets currently available to the agent. "
                 "DO NOT generate skill text that denies capability covered here."
        )
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    def __init__(self, skill_text: str):
        super().__init__()
        self.skill_text = skill_text
        # Discover toolsets once at module construction (cheap — just an AST parse).
        # Lazy-imported to keep skill_module.py free of evolution.core dependency
        # if anyone imports SkillModule standalone.
        from evolution.core.toolset_discovery import format_toolsets_for_prompt
        self._available_toolsets_prompt = format_toolsets_for_prompt()
        self.predictor = dspy.ChainOfThought(self.TaskWithSkill)

    def forward(self, task_input: str, available_toolsets: str | None = None) -> dspy.Prediction:
        # ``available_toolsets`` is accepted as a kwarg so DSPy's evaluator can
        # pass it through from with_inputs("task_input", "available_toolsets"),
        # but we IGNORE the supplied value and use the one captured at __init__
        # time. This keeps the toolsets ground truth identical across every
        # rollout in a run, regardless of how the example was constructed.
        result = self.predictor(
            skill_instructions=self.skill_text,
            task_input=task_input,
            available_toolsets=self._available_toolsets_prompt,
        )
        return dspy.Prediction(output=result.output)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
