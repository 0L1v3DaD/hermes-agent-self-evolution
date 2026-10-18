"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
        skill_dir: Optional[Path] = None,
    ) -> list[ConstraintResult]:
        """Run all applicable constraints. Returns list of results."""
        results = []

        # 1. Size limits
        results.append(self._check_size(artifact_text, artifact_type))

        # 2. Growth limit (if baseline provided)
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity & Harness² SurfaceSpec Rule 3 (script existence + py_compile)
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))
            if skill_dir is not None:
                results.append(self.validate_skill_scripts(artifact_text, skill_dir))

        # 5. Capability-denial guard (skill-only — protect agentic skills from
        #    optimizers that learn to "honestly refuse" rather than attempt the tool).
        #    Only enforced when baseline_text is provided so we can compare against it
        #    (the baseline may legitimately mention sandbox limits).
        if artifact_type == "skill" and baseline_text:
            results.append(self._check_no_capability_denial(artifact_text, baseline_text))

        # 6. Harness² SurfaceSpec Rule 2 (arXiv:2610.14224): no unescaped Jinja
        #    template braces ({{ or {%) outside fenced code blocks in always-on prompts.
        if artifact_type in ("system_prompt", "always_on", "prompt_prefix"):
            results.append(self._check_no_template_braces(artifact_text))

        return results

    @staticmethod
    def classify_edit_dimension(artifact_type: str) -> str:
        """Harness² Orthogonal Proposer Split (arXiv:2610.14224):
        Classifies an edit into 'INV' (invariant machinery: how the agent works)
        or 'DOM' (domain practitioner knowledge: skills, facts, domain workflows).
        """
        inv_types = {"tool_description", "param_description", "system_prompt", "always_on", "prompt_prefix", "code", "script", "hook"}
        return "INV" if artifact_type in inv_types else "DOM"

    @staticmethod
    def _check_no_template_braces(text: str) -> ConstraintResult:
        """Harness² SurfaceSpec Rule 2 (arXiv:2610.14224):
        Rejects raw Jinja2 template braces ({{ ... }} or {% ... %}) outside fenced
        markdown code blocks in always-on text.
        """
        import re

        stripped_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        matches = re.findall(r"(\{\{.*?\}\}|\{%.*?%\})", stripped_fences)
        if not matches:
            return ConstraintResult(
                passed=True,
                constraint_name="no_template_braces",
                message="No raw Jinja template braces in always-on text",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="no_template_braces",
            message=f"Found {len(matches)} unescaped Jinja template tag(s) in always-on text",
            details=", ".join(matches[:3]),
        )

    @staticmethod
    def validate_skill_scripts(skill_text: str, skill_dir: Optional[Path] = None) -> ConstraintResult:
        """Harness² SurfaceSpec Rule 3 (arXiv:2610.14224):
        Every helper script referenced as `scripts/<name>.py` in SKILL.md must exist
        inside the skill's `scripts/` directory and pass `py_compile`.
        """
        import py_compile
        import re

        script_refs = sorted(set(re.findall(r"\bscripts/([a-zA-Z0-9_\-]+\.py)\b", skill_text)))
        if not script_refs or skill_dir is None:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_scripts_valid",
                message=f"Skill script references checked ({len(script_refs)} ref(s))",
            )

        missing_or_invalid: list[str] = []
        for script_name in script_refs:
            script_path = skill_dir / "scripts" / script_name
            if not script_path.is_file():
                missing_or_invalid.append(f"missing: scripts/{script_name}")
                continue
            try:
                py_compile.compile(str(script_path), doraise=True)
            except py_compile.PyCompileError as exc:
                missing_or_invalid.append(f"syntax_error: scripts/{script_name} ({exc.msg})")

        if not missing_or_invalid:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_scripts_valid",
                message=f"All {len(script_refs)} referenced skill script(s) exist and compile",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="skill_scripts_valid",
            message=f"{len(missing_or_invalid)} referenced skill script(s) missing or invalid",
            details="; ".join(missing_or_invalid),
        )

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(hermes_repo),
            )

            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )
            else:
                # Extract failure summary
                last_lines = result.stdout.strip().split("\n")[-5:] if result.stdout else []
                return ConstraintResult(
                    passed=False,
                    constraint_name="test_suite",
                    message="Test suite failed",
                    details="\n".join(last_lines),
                )
        except subprocess.TimeoutExpired:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite timed out (300s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        size = len(text)
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        elif artifact_type == "prompt_prefix":
            # PATCH 5 (2026-05-10) — added for additive reassembly tier-1 validation.
            # The GEPA-evolved prefix block (typically 1-15K chars) gets size-checked
            # against this limit; the FULL merged artifact is checked against
            # max_skill_size separately.
            limit = self.config.max_prompt_prefix_size
        else:
            limit = self.config.max_skill_size  # Default

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="size_limit",
                message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
            )

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        # PATCH 5 (2026-05-10): prompt_prefix uses a much larger growth budget
        # because GEPA's whole purpose is to GROW the instruction prefix from
        # ~500 baseline chars to multi-KB behavioral guidance. The 20% cap
        # appropriate for skill-body growth would force the optimizer to
        # output near-empty prefixes, defeating optimization.
        if artifact_type == "prompt_prefix":
            max_growth = self.config.max_prefix_growth
        else:
            max_growth = self.config.max_prompt_growth

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="growth_limit",
                message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
            )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        has_frontmatter = text.strip().startswith("---")
        has_name = "name:" in text[:500] if has_frontmatter else False
        has_description = "description:" in text[:500] if has_frontmatter else False

        if has_frontmatter and has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        else:
            missing = []
            if not has_frontmatter:
                missing.append("YAML frontmatter (---)")
            if not has_name:
                missing.append("name field")
            if not has_description:
                missing.append("description field")
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill missing: {', '.join(missing)}",
            )

    # Pattern set chosen empirically from a real failed evolved artifact
    # (2026-05-10 evolved hallucinated capability denial). Each pattern matched
    # the offending output but produced 0 false positives on the baseline skill.
    _CAPABILITY_DENIAL_PATTERNS = [
        # Surface-form headers the optimizer learned to emit
        (r"##\s*(your\s+)?operational\s+constraint", "section header 'Operational Constraints'"),
        (r"##\s*your\s+limitation", "section header 'Your Limitations'"),
        (r"##\s*what\s+you\s+(do\s+not|cannot|can'?t)", "section header denying capability"),
        # Direct capability-denial sentences (broad enough to catch paraphrases)
        (r"you\s+do\s+\*?\*?not\*?\*?\s+have\s+(live\s+)?access\s+to", "'you do not have access to'"),
        (r"you\s+(do\s+not|don'?t)\s+have\s+(the\s+)?(ability|access|tools?|capability)\s+to", "'you don't have the ability to'"),
        (r"you\s+are\s+(a\s+)?(text-based|stateless)\s+(llm|model|ai)\s+processing", "'you are a text-based LLM processing...' frame"),
        (r"you\s+cannot\s+(read|write|access|execute|run|browse|fetch|delegate|spawn)", "'you cannot <action>' assertion"),
        # The specific bullet-list shape the run produced
        (r"-\s+(the\s+user'?s?\s+)?(local\s+filesystem|live\s+web\s+brows|gcp/cloud\s+credentials)", "capability-denial bullet item"),
    ]

    def _check_no_capability_denial(self, text: str, baseline: str) -> ConstraintResult:
        """Reject evolved artifacts that hallucinate capability denial.

        The bias we are guarding against: the GEPA optimizer sometimes generalizes
        from a few sub-agent / sandboxed examples in the dataset and writes new text
        like "## Your Operational Constraints — you do NOT have live access to ...".
        That text is a falsehood for the full Hermes Agent (which has filesystem,
        terminal, browser, delegation tools available) and degrades behavior by
        teaching the agent to refuse instead of attempt.

        Strategy: regex against patterns that DID appear in a confirmed-bad evolved
        artifact AND did NOT appear in the trusted baseline. We only flag patterns
        that are NEW in the evolved (i.e. not present in baseline) — that prevents
        false positives if the baseline itself legitimately discusses a real sandbox.
        """
        import re

        baseline_lower = baseline.lower()
        text_lower = text.lower()

        new_violations: list[tuple[int, str, str]] = []
        for pattern, label in self._CAPABILITY_DENIAL_PATTERNS:
            evolved_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if not evolved_matches:
                continue
            baseline_matches = len(re.findall(pattern, baseline_lower, re.IGNORECASE))
            # Only flag NEW occurrences — if baseline already had this many, that's pre-existing.
            if len(evolved_matches) > baseline_matches:
                # Capture line text of first new match for evidence
                first_match = evolved_matches[baseline_matches]
                line_no = text_lower.count("\n", 0, first_match.start()) + 1
                line = text.split("\n")[line_no - 1].strip()[:160]
                new_violations.append((line_no, label, line))

        if not new_violations:
            return ConstraintResult(
                passed=True,
                constraint_name="no_capability_denial",
                message="No capability-denial patterns introduced vs baseline",
            )

        evidence = "; ".join(
            f"L{ln} [{label}]: {line!r}" for ln, label, line in new_violations[:3]
        )
        return ConstraintResult(
            passed=False,
            constraint_name="no_capability_denial",
            message=f"Evolved introduced {len(new_violations)} capability-denial pattern(s) absent from baseline",
            details=evidence,
        )
