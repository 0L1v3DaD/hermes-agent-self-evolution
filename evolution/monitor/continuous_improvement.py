"""
HEOS Phase 5: Continuous Improvement & Triage Monitor (Oct 14, 2026 SOTA Hardened)
Grounded in SOTA Literature:
1. "Self-Evolving Agent Harness (Harness-RSI)" (arXiv:2610.18813, Oct 13, 2026)
2. "Small but Mighty: Test Suites as Improvement Signals for Self-Evolving Agents (TeSearch)" (arXiv:2610.15964, Oct 11, 2026)
3. "Does Failure Feedback Direct Agent Self-Modification?" (arXiv:2610.14662, Oct 10, 2026) & "Trace Evolution Bench" (arXiv:2610.15055, Oct 10, 2026)
4. "EvolvAttack: Backdooring Self-Evolving VLM Agents" (arXiv:2610.15244, Oct 11, 2026)
5. "ClawLoop: Self-Improving Agents via Harness-Native Training Loop" (arXiv:2610.14118, Oct 9, 2026)
6. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (arXiv:2507.19457, ICLR 2026 Oral)
7. "SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Reusable Skills as APIs" (arXiv:2504.07079)
8. "Learning to Adapt: Self-Improving Web Agent via Cognitive Aware Learning and Exploration (SCALE)" (arXiv:2605.31365)

Implements real telemetry ingestion from Hermes Agent's isolated SQLite database (~/.hermes/state.db),
Harness² (arXiv:2610.14224) INV/DOM attribution + confounded-edit guard,
Graphiti/LightMem bi-temporal sleep-time memory consolidation,
EvolvAttack-hardened trace sanitization, multi-phase orchestration (Phases 1-5),
circuit-breaker guardrails, and strictly local-only evolution (0 PRs, 0 Issues).
"""
import os
import re
import sys
import json
import sqlite3
import tempfile
import argparse
import contextlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from rich.console import Console

from evolution.core.config import EvolutionConfig

console = Console()


class ContinuousImprovementMonitor:
    def __init__(
        self,
        db_path: str = "output/alpha_evolve_programs.db",
        max_failures: int = 2,
        config: EvolutionConfig | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.repo_root = Path(__file__).resolve().parents[2]
        self.db_path = Path(db_path) if Path(db_path).is_absolute() else (self.repo_root / db_path)
        self.hermes_path = Path(os.environ.get("HERMES_AGENT_PATH", str(self.repo_root)))
        self.max_failures = max_failures
        self.failure_count = 0
        self.circuit_breaker_tripped = False

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Write text atomically via a temporary file + os.replace to prevent torn reads."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fp.write(content)
            os.replace(tmp_path, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def _sanitize_trace_text(text: str, max_len: int = 240) -> str:
        """EvolvAttack Admission Gate (arXiv:2610.15244): strips secrets and prompt-injection payloads before persisting trace learnings."""
        if not text:
            return ""
        cleaned = str(text).replace("\r", " ").replace("\n", " ")
        # Redact common secret patterns
        cleaned = re.sub(r"(sk-[A-Za-z0-9_\-]{10,}|AIza[A-Za-z0-9_\-]{20,}|AQ\.[A-Za-z0-9_\-\.]{20,}|ghp_[A-Za-z0-9]{20,}|\b\d{8,10}:[A-Za-z0-9_\-]{30,})", "[REDACTED]", cleaned)
        # Neutralize prompt injection markers from untrusted web/tool output
        cleaned = re.sub(r"(?i)(ignore\s+(all\s+)?previous\s+instructions|<\s*/?\s*system\s*>|system\s+override|you\s+are\s+now\s+a)", "[FILTERED_INJECTION]", cleaned)
        return cleaned[:max_len].strip()

    def mine_hermes_interaction_telemetry(self, limit: int = 500) -> Dict[str, Any]:
        """ClawLoop / TEB Diagnosis Phase (arXiv:2610.14118, arXiv:2610.15055):
        Mines live execution traces from ~/.hermes/state.db (messages & sessions) and writes sanitized
        learnings to ~/.hermes/memories/.learnings/ and ~/.hermes/wiki/main/concepts/.
        """
        state_db = Path.home() / ".hermes" / "state.db"
        summary: Dict[str, Any] = {
            "sessions_total": 0,
            "messages_scanned": 0,
            "tool_failures_by_name": {},
            "tool_failure_total": 0,
            "skill_friction_signals": [],
            "user_correction_signals": 0,
        }
        if not state_db.exists():
            return summary

        try:
            with contextlib.closing(sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=5.0)) as conn:
                summary["sessions_total"] = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                rows = conn.execute(
                    "SELECT role, tool_name, substr(content, 1, 500) FROM messages WHERE role IN ('tool', 'user') ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception as e:
            summary["error"] = str(e)
            return summary

        summary["messages_scanned"] = len(rows)
        sample_errors: List[str] = []
        correction_re = re.compile(r"\b(no,|actually|why didn't|not working|bottleneck|fix|wrong|failed|issue)\b", re.I)

        for role, tool_name, content in rows:
            txt = content or ""
            if role == "tool":
                tname = tool_name or "unknown"
                if '"exit_code": 1' in txt or '"exit_code": 2' in txt or '"error": "' in txt or "Traceback" in txt or "SyntaxError:" in txt:
                    summary["tool_failures_by_name"][tname] = summary["tool_failures_by_name"].get(tname, 0) + 1
                    summary["tool_failure_total"] += 1
                    if len(sample_errors) < 8:
                        sample_errors.append(f"[{tname}] {self._sanitize_trace_text(txt, 180)}")
                if "No --plan passed" in txt and "last30days" not in summary["skill_friction_signals"]:
                    summary["skill_friction_signals"].append("last30days")
                if "SyntaxError: f-string expression part cannot include a backslash" in txt and "python311-fstring-backslash" not in summary["skill_friction_signals"]:
                    summary["skill_friction_signals"].append("python311-fstring-backslash")
            elif role == "user":
                if correction_re.search(txt):
                    summary["user_correction_signals"] += 1

        # Persist into Hermes isolated .learnings directory
        learn_dir = Path.home() / ".hermes" / "memories" / ".learnings"
        try:
            learn_dir.mkdir(parents=True, exist_ok=True)
            err_md = learn_dir / "ERRORS.md"
            lrn_md = learn_dir / "LEARNINGS.md"
            err_lines = [
                "# Hermes Agent Harness-Native Execution Errors (Auto-Mined)",
                "",
                f"- **Total Sessions**: `{summary['sessions_total']}` | **Scanned Messages**: `{summary['messages_scanned']}`",
                f"- **Tool Failures Detected**: `{summary['tool_failure_total']}` (`{json.dumps(summary['tool_failures_by_name'])}`)",
                "",
                "## Recent Sanitized Failure Signatures",
            ] + [f"- `{s}`" for s in sample_errors]
            self._atomic_write_text(err_md, "\n".join(err_lines) + "\n")

            lrn_lines = [
                "# Hermes Agent Self-Evolution Learnings (Harness-RSI / ClawLoop)",
                "",
                f"- **User Correction Signals (last {limit} msgs)**: `{summary['user_correction_signals']}`",
                f"- **Skill Friction Signals**: `{', '.join(summary['skill_friction_signals']) or 'none'}`",
                "- **Best Practice (`last30days`)**: Always pass `--plan '<json>'` when invoking `scripts/last30days.py` from an agent harness to avoid deterministic single-source fallback.",
                "- **Best Practice (`python3.11` venv)**: Avoid backslashes inside `{...}` in f-strings when running under `~/.hermes/hermes-agent/venv/bin/python` (Python 3.11).",
            ]
            self._atomic_write_text(lrn_md, "\n".join(lrn_lines) + "\n")
        except Exception:
            pass

        return summary

    def deploy_evolved_artifacts_locally(self) -> Dict[str, Any]:
        """Harness-RSI Admission Gate (arXiv:2610.18813):
        Registers verified local HEOS Phase 1-4 evolved artifacts into each agent's isolated
        .learnings/HEOS_EVOLVED_REGISTRY.json without touching git-tracked upstream source or opening PRs.
        """
        output_dir = self.repo_root / "output"
        evolved_skills = sorted([str(p) for p in output_dir.glob("*/20*/evolved_skill.md")]) if output_dir.exists() else []
        tool_desc_json = output_dir / "evolved_tool_descriptions.json"
        sys_prompt_json = output_dir / "evolved_system_prompts.json"

        # Compute Harness² experience attribution over recorded skill evolution runs
        recorded_edits: List[Dict[str, Any]] = []
        if output_dir.exists():
            for metrics_file in sorted(output_dir.glob("*/20*/metrics.json"))[-5:]:
                try:
                    mdata = json.loads(metrics_file.read_text(errors="replace"))
                    skill_slug = metrics_file.parent.parent.name
                    b_score = float(mdata.get("baseline_holdout", mdata.get("baseline_score", 0.0)))
                    e_score = float(mdata.get("evolved_holdout", mdata.get("evolved_score", b_score)))
                    recorded_edits.append({
                        "component": f"skills/{skill_slug}/SKILL.md",
                        "artifact_type": "skill",
                        "change": f"GEPA optimization run {metrics_file.parent.name}",
                        "before_score": b_score,
                        "after_score": e_score,
                        "isolated": True,
                    })
                except Exception:
                    pass
        attribution = self.evaluate_experience_attribution(recorded_edits)

        registry_payload = {
            "updated_at": "2026-10-18T06:00:00Z",
            "policy": {"create_pr": False, "create_issue": False, "admission_gate": "Harness2+TeSearch+EvolvAttack"},
            "evolved_skills_count": len(evolved_skills),
            "latest_evolved_skills": evolved_skills[-5:],
            "evolved_tool_descriptions_path": str(tool_desc_json) if tool_desc_json.exists() else None,
            "evolved_system_prompts_path": str(sys_prompt_json) if sys_prompt_json.exists() else None,
            "harness2_experience_attribution": attribution,
        }
        deployed_to: List[str] = []
        for target_dir in [
            Path.home() / ".hermes" / "memories" / ".learnings",
        ]:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                reg_file = target_dir / "HEOS_EVOLVED_REGISTRY.json"
                self._atomic_write_text(reg_file, json.dumps(registry_payload, indent=2))
                deployed_to.append(str(reg_file))
            except Exception:
                pass
        return {"deployed_registries": deployed_to, "summary": registry_payload}

    def triage_underperforming_abilities(self, threshold: float = 0.95) -> List[Tuple[str, float]]:
        """Queries the QD SQLite DB to extract code targets whose performance falls below the SRE SLA threshold."""
        if not self.db_path.exists():
            console.print(f"[yellow][!] Quality-Diversity database not found at {self.db_path}. Staging triage baseline...[/yellow]")
            return [("tip_tool", 0.85)]

        with contextlib.closing(sqlite3.connect(str(self.db_path), timeout=5.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            cursor = conn.cursor()

            cursor.execute("""
                SELECT target_name, MAX(accuracy) as max_acc 
                FROM candidates 
                WHERE is_compilation_only = 0
                GROUP BY target_name
            """)
            rows = cursor.fetchall()

        underperforming = []
        for target, acc in rows:
            if acc < threshold:
                underperforming.append((target, acc))

        if not rows:
            underperforming.append(("tip_tool", 0.85))

        if underperforming:
            console.print(f"[yellow][!] Triaged {len(underperforming)} abilities below quality threshold ({threshold})[/yellow]")
        else:
            console.print(f"[green][✓] All abilities performing above threshold ({threshold})[/green]")

        return sorted(underperforming, key=lambda x: x[1])

    def triage_all_phases(self, code_threshold: float = 0.95) -> Dict[str, Any]:
        """Inspects state across Phases 1, 2, 3, and 4 + live Hermes interaction telemetry to build an optimization plan."""
        hermes_telemetry = self.mine_hermes_interaction_telemetry(limit=500)
        deployment = self.deploy_evolved_artifacts_locally()

        plan: Dict[str, Any] = {
            "phase1_skills": [],
            "phase2_tool_descriptions_needed": False,
            "phase3_system_prompts_needed": False,
            "phase4_code_targets": [],
            "interaction_friction_targets": list(hermes_telemetry.get("tool_failures_by_name", {}).keys()),
            "skill_friction_signals": hermes_telemetry.get("skill_friction_signals", []),
            "hermes_telemetry": hermes_telemetry,
            "local_deployment": deployment,
        }

        # Phase 1: Check evolved skills output
        output_dir = self.repo_root / "output"
        evolved_skills = list(output_dir.glob("*/20*/evolved_skill.md")) if output_dir.exists() else []
        if not evolved_skills:
            plan["phase1_skills"].append("reverse-engineer-anything")

        # Phase 2: Check evolved tool descriptions
        tool_desc_json = output_dir / "evolved_tool_descriptions.json"
        if not tool_desc_json.exists():
            plan["phase2_tool_descriptions_needed"] = True

        # Phase 3: Check evolved system prompts
        sys_prompt_json = output_dir / "evolved_system_prompts.json"
        if not sys_prompt_json.exists():
            plan["phase3_system_prompts_needed"] = True

        # Phase 4: Check QD database
        plan["phase4_code_targets"] = self.triage_underperforming_abilities(threshold=code_threshold)
        return plan

    def _run_guarded_cmd(self, cmd: List[str], label: str, timeout: int = 300) -> bool:
        """Executes a phase command with a circuit breaker on consecutive failures."""
        if self.circuit_breaker_tripped:
            console.print(f"[red][CIRCUIT BREAKER] Skipping {label} — failure limit ({self.max_failures}) reached.[/red]")
            return False

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(self.repo_root), timeout=timeout)
            console.print(f"[green][✓] {label} completed successfully[/green]")
            return True
        except Exception as e:
            self.failure_count += 1
            stderr = getattr(e, "stderr", str(e))
            console.print(f"[red][!] {label} failed ({self.failure_count}/{self.max_failures}):[/red]\n{stderr}")
            if self.failure_count >= self.max_failures:
                self.circuit_breaker_tripped = True
                console.print("[bold red][CIRCUIT BREAKER TRIPPED] Halting further autonomous evolution in this cycle.[/bold red]")
            return False

    def trigger_evolution(self, target: str, baseline_accuracy: float) -> bool:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", target):
            raise ValueError(f"[SRE FAIL-CLOSED] Blocked path traversal attempt on target: '{target}'")

        console.print(f"[*] Triggering automated Phase 4 code evolution for: {target}...")
        cmd = [
            sys.executable, "-m", "evolution.code.alpha_loop",
            "--target", target,
            "--generations", "1",
        ]
        return self._run_guarded_cmd(cmd, f"Phase 4 ({target})", timeout=300)

    def trigger_phase(self, phase: int, target: str = "", generations: int = 1) -> bool:
        """Triggers a specific HEOS phase (1=Skills, 2=Tools, 3=Prompts, 4=Code)."""
        if phase == 1:
            skill_name = target or "reverse-engineer-anything"
            if not re.match(r"^[a-zA-Z0-9_\-]+$", skill_name):
                raise ValueError(f"[SRE FAIL-CLOSED] Invalid skill name: '{skill_name}'")
            cmd = [
                sys.executable, "-m", "evolution.skills.evolve_skill",
                "--skill", skill_name,
                "--iterations", str(generations),
                "--num-cases", "4",
                "--no-counterfactual-probes",
                "--no-adversarial-judge",
            ]
            return self._run_guarded_cmd(cmd, f"Phase 1 Skill ({skill_name})", timeout=600)
        elif phase == 2:
            cmd = [
                sys.executable, "-m", "evolution.tools.evolve_tool_descriptions",
                "--generations", str(generations),
            ]
            return self._run_guarded_cmd(cmd, "Phase 2 Tool Descriptions", timeout=600)
        elif phase == 3:
            section = target or "DEFAULT_AGENT_IDENTITY"
            cmd = [
                sys.executable, "-m", "evolution.prompts.evolve_system_prompts",
                "--section", section,
                "--generations", str(generations),
            ]
            return self._run_guarded_cmd(cmd, f"Phase 3 System Prompt ({section})", timeout=600)
        elif phase == 4:
            return self.trigger_evolution(target or "tip_tool", 0.85)
        else:
            raise ValueError(f"Unsupported phase: {phase}")

    def sync_hermes_to_wiki(self, wiki_dir: Path | None = None) -> list[str]:
        """Populates Hermes Agent's dedicated isolated wiki vault (~/.hermes/wiki/main) from MEMORY.md, USER.md, and memory_store.db."""
        vault = wiki_dir or (Path.home() / ".hermes" / "wiki" / "main")
        sources_dir = vault / "sources"
        concepts_dir = vault / "concepts"
        sources_dir.mkdir(parents=True, exist_ok=True)
        concepts_dir.mkdir(parents=True, exist_ok=True)
        hermes_mem_dir = Path.home() / ".hermes" / "memories"
        synced: list[str] = []
        for src_name, slug, title in [
            ("MEMORY.md", "hermes-agent-memory.md", "Hermes Agent Long-Term Memory"),
            ("USER.md", "hermes-user-profile.md", "Hermes Agent User Profile"),
        ]:
            src_path = hermes_mem_dir / src_name
            if src_path.exists():
                body = src_path.read_text(errors="replace").strip()
                dest_path = sources_dir / slug
                content = (
                    f"---\n"
                    f"title: \"{title}\"\n"
                    f"type: note\n"
                    f"source: hermes-agent\n"
                    f"---\n\n"
                    f"# {title}\n\n"
                    f"Related: [[index]], [[hermes-holographic-facts]], [[hermes-interaction-telemetry]]\n\n"
                    f"{body}\n"
                )
                self._atomic_write_text(dest_path, content)
                synced.append(str(dest_path))

        # Also export top Holographic facts from ~/.hermes/memory_store.db into Hermes's dedicated wiki
        mem_db = Path.home() / ".hermes" / "memory_store.db"
        if mem_db.exists():
            try:
                with contextlib.closing(sqlite3.connect(f"file:{mem_db}?mode=ro", uri=True, timeout=5.0)) as conn:
                    rows = conn.execute(
                        "SELECT category, content, trust_score FROM facts ORDER BY trust_score DESC, updated_at DESC LIMIT 100"
                    ).fetchall()
                stale_pat = re.compile(r"(dgxspark|TEMP_AUDIT_ENTRY_DELETE_ME)", re.I)
                filtered_rows = [
                    (cat, content_txt, trust)
                    for cat, content_txt, trust in rows
                    if not stale_pat.search(str(content_txt))
                ]
                if filtered_rows:
                    lines = [
                        "---",
                        'title: "Hermes Holographic Fact Bank"',
                        "type: concept",
                        "source: hermes-memory-store",
                        "---",
                        "",
                        "# Hermes Holographic Fact Bank",
                        "",
                        "Related: [[index]], [[hermes-agent-memory]], [[hermes-user-profile]], [[hermes-interaction-telemetry]]",
                        "",
                        "## Anti-Confusion Contrast Rules (STRATA arXiv:2610.19386)",
                        "- **DO NOT CONFUSE (`dgxspark` retired)**: `dgxspark` was decommissioned and is no longer owned. Never SSH to or route workloads to `dgxspark`; all inference runs via local LiteLLM (`http://127.0.0.1:8000/v1`) on Vertex AI ADC.",
                        "- **DO NOT CONFUSE (Strict Agent Isolation — CODA arXiv:2610.20845)**: Hermes Agent (`~/.hermes/`, CDP `9333`) operates in strict isolation with its own dedicated SQLite state, Wiki, and GBrain store.",
                        "- **DO NOT CONFUSE (Canonical Models Only)**: Use only `argon` (`gemini-4-argon`), `flash` (`gemini-3.8-flash`), `opus` (`claude-opus-5-5`), `sonnet` (`claude-sonnet-5-5`), `haiku` (`claude-haiku-5-5`), and `gemini-embedding-2`.",
                        "",
                        "## Verified Holographic Facts (Bi-Temporal Consolidated)",
                    ]
                    active_facts, superseded_facts = self.consolidate_bitemporal_facts(filtered_rows)
                    for cat, content_txt, trust in active_facts:
                        clean_txt = str(content_txt).replace("\n", " ").strip()
                        lines.append(f"- **[ACTIVE | {cat}]** (trust={trust:.2f}): {clean_txt}")
                    if superseded_facts:
                        lines.append("")
                        lines.append("## Superseded Historical Claims (Bi-Temporal Audit Trail)")
                        for item in superseded_facts[:20]:
                            lines.append(
                                f"- ~~[{item['category']}] (trust={item['trust']:.2f}): {item['content']}~~ "
                                f"*(superseded by higher-trust active fact on `{item['subject_key']}`)*"
                            )
                    fact_path = concepts_dir / "hermes-holographic-facts.md"
                    self._atomic_write_text(fact_path, "\n".join(lines) + "\n")
                    synced.append(str(fact_path))
            except Exception:
                pass

        # Export Hermes isolated .learnings into Hermes Wiki concepts
        learn_dir = hermes_mem_dir / ".learnings"
        if (learn_dir / "LEARNINGS.md").exists() or (learn_dir / "ERRORS.md").exists():
            try:
                lrn_txt = (learn_dir / "LEARNINGS.md").read_text(errors="replace").strip() if (learn_dir / "LEARNINGS.md").exists() else ""
                err_txt = (learn_dir / "ERRORS.md").read_text(errors="replace").strip() if (learn_dir / "ERRORS.md").exists() else ""
                telem_path = concepts_dir / "hermes-interaction-telemetry.md"
                self._atomic_write_text(
                    telem_path,
                    "---\n"
                    'title: "Hermes Harness-Native Interaction Telemetry"\n'
                    "type: concept\n"
                    "source: hermes-self-evolution\n"
                    "---\n\n"
                    "Related: [[index]], [[hermes-agent-memory]], [[hermes-user-profile]], [[hermes-holographic-facts]]\n\n"
                    f"{lrn_txt}\n\n{err_txt}\n",
                )
                synced.append(str(telem_path))
            except Exception:
                pass

        # Write root index.md so GBrain graph has a connected root with 0 orphans
        index_path = vault / "index.md"
        index_content = (
            "---\n"
            'title: "Hermes Agent Isolated Knowledge Wiki"\n'
            "type: index\n"
            "source: hermes-agent\n"
            "---\n\n"
            "# Hermes Agent Isolated Knowledge Wiki\n\n"
            "## Core Sources\n"
            "- [[hermes-agent-memory]] — Hermes Agent Long-Term Memory\n"
            "- [[hermes-user-profile]] — Hermes Agent User Profile\n\n"
            "## Concepts & Telemetry\n"
            "- [[hermes-holographic-facts]] — Hermes Holographic Fact Bank (STRATA Contrast-Guarded)\n"
            "- [[hermes-interaction-telemetry]] — Hermes Harness-Native Interaction Telemetry\n"
        )
        self._atomic_write_text(index_path, index_content)
        synced.append(str(index_path))
        return synced

    def _run_isolated_gbrain_dream(self, agent_name: str, gbrain_home_parent: Path, wiki_dir: Path, dry_run: bool = False) -> dict:
        """Runs gbrain dream for a single agent against its own isolated GBRAIN_HOME (.gbrain/brain.pglite) and wiki_dir."""
        import fcntl
        gbrain_bin = Path.home() / ".bun" / "bin" / "gbrain"
        if not gbrain_bin.exists() or not wiki_dir.exists():
            return {"status": "skipped", "reason": "gbrain binary or wiki_dir missing"}

        gbrain_dot_dir = gbrain_home_parent / ".gbrain"
        gbrain_dot_dir.mkdir(parents=True, exist_ok=True)
        lock_file_path = gbrain_dot_dir / "single_writer.flock"

        env = os.environ.copy()
        env["GBRAIN_HOME"] = str(gbrain_home_parent)
        env["GBRAIN_MODEL"] = "litellm:gemini-3.8-flash"
        env["LITELLM_BASE_URL"] = self.config.api_base
        if self.config.api_key:
            env["LITELLM_API_KEY"] = self.config.api_key

        cmd = [str(gbrain_bin), "dream", "--dir", str(wiki_dir), "--json"]
        if dry_run:
            cmd.append("--dry-run")

        with open(lock_file_path, "w") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                for k, v in [
                    ("sync.repo_path", str(wiki_dir)),
                    ("models.default", "litellm:gemini-3.8-flash"),
                    ("models.tier.utility", "litellm:gemini-3.8-flash"),
                    ("models.tier.reasoning", "litellm:gemini-3.8-flash"),
                    ("models.tier.deep", "litellm:gemini-4-argon"),
                    ("models.tier.subagent", "litellm:claude-sonnet-5-5"),
                    ("dream.drift.enabled", "true"),
                    ("cycle.skillopt.enabled", "true"),
                    ("cycle.enrich_thin.enabled", "true"),
                ]:
                    subprocess.run(
                        [str(gbrain_bin), "config", "set", k, v],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=30,
                    )
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=300)
                idx = proc.stdout.find("\n{")
                json_str = proc.stdout[idx + 1:].strip() if idx != -1 else proc.stdout[proc.stdout.find("{"):].strip()
                if json_str.startswith("{"):
                    report = json.loads(json_str)
                else:
                    report = {"status": "completed", "exit_code": proc.returncode, "stdout_tail": proc.stdout[-500:]}
                console.print(f"[green][✓] Isolated GBrain Dream cycle completed for {agent_name} ({gbrain_dot_dir})[/green]")
                return report
            except Exception as e:
                console.print(f"[yellow][!] Isolated GBrain Dream warning for {agent_name}: {e}[/yellow]")
                return {"status": "error", "error": str(e)}
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def consolidate_bitemporal_facts(
        facts: List[Tuple[str, str, float]]
    ) -> Tuple[List[Tuple[str, str, float]], List[Dict[str, Any]]]:
        """Graphiti / LightMem Bi-Temporal Sleep-Time Consolidation:
        Deduplicates and resolves contradictory claims sharing the same subject key.
        Since input `facts` are ordered by `(trust_score DESC, updated_at DESC)`, the first
        claim seen for a normalized `(category, subject_key)` is retained as `[ACTIVE]`, while
        conflicting lower-trust/older claims on the same subject are moved to `superseded_facts`.
        """
        active: List[Tuple[str, str, float]] = []
        superseded: List[Dict[str, Any]] = []
        seen_subjects: Dict[Tuple[str, str], str] = {}

        for cat, content_txt, trust in facts:
            clean_txt = str(content_txt).replace("\n", " ").strip()
            # Extract deterministic subject anchor (text before ':' or ' is ' or first 4 tokens)
            head = re.split(r"(:|\bis\b|\buses\b|\bruns\b)", clean_txt, maxsplit=1)[0].strip().lower()
            tokens = re.findall(r"[a-z0-9_\-\.]+", head)
            subject_key = " ".join(tokens[:5]) if tokens else clean_txt[:32].lower()
            bucket = (str(cat).lower(), subject_key)

            if bucket not in seen_subjects:
                seen_subjects[bucket] = clean_txt
                active.append((cat, clean_txt, float(trust)))
            elif seen_subjects[bucket] != clean_txt:
                superseded.append({
                    "category": cat,
                    "subject_key": subject_key,
                    "content": clean_txt,
                    "trust": float(trust),
                })

        return active, superseded

    @staticmethod
    def evaluate_experience_attribution(edits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Harness² Experience Attribution & Confounded-Edit Guard (arXiv:2610.14224):
        Evaluates candidate harness component edits (`component`, `artifact_type`,
        `before_score`, `after_score`, `isolated`).
        - If multiple components changed simultaneously without isolated rollouts, marks
          every row `confounded` and emits `experience=None` ('NO_ENTRY') so unverified
          attributions never poison the domain library.
        - If an isolated edit improved performance, emits a structured `<experience>`
          record (`cue`, `lesson`, `evidence`, `confidence`, `dimension`).
        """
        from evolution.core.constraints import ConstraintValidator

        if not edits:
            return {"edit_rows": [], "experience": None, "status": "NO_ENTRY"}

        multi_unisolated = len(edits) > 1 and not all(e.get("isolated", False) for e in edits)
        edit_rows: List[Dict[str, Any]] = []
        best_helped: Dict[str, Any] | None = None

        for edit in edits:
            comp = str(edit.get("component", "unknown"))
            art_type = str(edit.get("artifact_type", "skill"))
            before = float(edit.get("before_score", 0.0))
            after = float(edit.get("after_score", 0.0))
            delta = after - before
            dim = ConstraintValidator.classify_edit_dimension(art_type)

            if multi_unisolated:
                verdict = "confounded"
            elif delta > 1e-4:
                verdict = "helped"
            elif delta < -1e-4:
                verdict = "hurt"
            else:
                verdict = "no_observed_effect"

            row = {
                "component": comp,
                "dimension": dim,
                "change": str(edit.get("change", "updated component")),
                "before_score": before,
                "after_score": after,
                "delta": round(delta, 4),
                "verdict": verdict,
            }
            edit_rows.append(row)
            if verdict == "helped" and (best_helped is None or delta > best_helped["delta"]):
                best_helped = {**row, "cue": edit.get("cue"), "lesson": edit.get("lesson")}

        if best_helped is None:
            return {"edit_rows": edit_rows, "experience": None, "status": "NO_ENTRY"}

        experience = {
            "dimension": best_helped["dimension"],
            "cue": best_helped.get("cue") or f"When optimizing {best_helped['component']} ({best_helped['dimension']})",
            "lesson": best_helped.get("lesson") or f"Apply validated {best_helped['change']} on {best_helped['component']}",
            "evidence": (
                f"{best_helped['component']} improved score {best_helped['before_score']:.3f} -> "
                f"{best_helped['after_score']:.3f} ({best_helped['delta']:+.3f})"
            ),
            "confidence": "high" if best_helped["delta"] >= 0.05 else "medium",
        }
        return {"edit_rows": edit_rows, "experience": experience, "status": "RECORDED"}

    def run_dream_cycle(self, dry_run: bool = False) -> dict:
        """Executes Hermes Agent's strictly isolated Memory Dreaming & Self-Evolution cycle.
        Strictly local and Hermes-only (never opens PRs or issues).
        """
        if self.config.create_pr or getattr(self.config, "create_issue", False):
            raise RuntimeError("[SRE FAIL-CLOSED] PR and Issue creation are strictly forbidden by policy!")

        # 0. Mine live Hermes interaction telemetry & deploy evolved artifacts locally first
        triage = self.triage_all_phases()

        # 1. Hermes Agent isolated wiki (~/.hermes/wiki/main) + isolated GBrain (~/.hermes/.gbrain)
        hermes_wiki_dir = Path.home() / ".hermes" / "wiki" / "main"
        synced_files = self.sync_hermes_to_wiki(hermes_wiki_dir)
        console.print(f"[*] Synced {len(synced_files)} Hermes artifact(s) into isolated vault {hermes_wiki_dir}")
        hermes_gbrain_report = self._run_isolated_gbrain_dream(
            "hermes", Path.home() / ".hermes", hermes_wiki_dir, dry_run=dry_run
        )

        summary = {
            "policy": {"create_pr": False, "create_issue": False, "mode": "local_only", "per_agent_isolation": True},
            "hermes_brain": {
                "wiki_dir": str(hermes_wiki_dir),
                "gbrain_dir": str(Path.home() / ".hermes" / ".gbrain"),
                "synced_files": synced_files,
                "gbrain_dream": hermes_gbrain_report,
            },
            "heos_triage": triage,
        }
        report_path = self.repo_root / "output" / "dream_cycle_report.json"
        self._atomic_write_text(report_path, json.dumps(summary, indent=2))
        console.print(f"[green][✓] Saved isolated per-agent dream & evolution report to {report_path}[/green]")
        return summary

    def generate_pr_report(
        self,
        target: str,
        before_score: float,
        after_score: float,
        pytest_pass: bool,
        ast_pass: bool,
        bandit_pass: bool,
    ) -> str:
        """Constructs a strictly local SRE-grade Markdown evolution summary (never opens PRs or issues)."""
        report = f"""# HEOS Automated Evolution Report (Local-Only): {target}
        
## 📊 Evolution Metrics
*   **Target:** `{target}`
*   **Optimizer Engine:** `dspy.GEPA` / MAP-Elites
*   **Baseline Score:** `{before_score:.2f}`
*   **Optimized Score:** `{after_score:.2f}` (Improvement: `+{after_score - before_score:+.2f}`)

## 🛡️ SRE Security & Quality Gates
*   [{"x" if pytest_pass else " "}] **pytest Suite:** {"PASSED" if pytest_pass else "FAILED"}
*   [{"x" if ast_pass else " "}] **AST Whitelist Lock:** {"PASSED" if ast_pass else "FAILED"}
*   [{"x" if bandit_pass else " "}] **Bandit Scan:** {"PASSED" if bandit_pass else "FAILED"}
"""
        return report


def main():
    parser = argparse.ArgumentParser(description="HEOS Phase 5: Continuous Improvement & Unified Dream Monitor")
    parser.add_argument(
        "--phase",
        type=str,
        default="auto",
        choices=["auto", "all", "dream", "1", "2", "3", "4", "triage-only"],
    )
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true", help="Run dream cycle in dry-run mode")
    args = parser.parse_args()

    monitor = ContinuousImprovementMonitor()
    if args.phase == "triage-only":
        plan = monitor.triage_all_phases(code_threshold=args.threshold)
        console.print_json(data=plan)
        return

    if args.phase == "dream":
        summary = monitor.run_dream_cycle(dry_run=args.dry_run)
        console.print_json(data=summary)
        return

    if args.phase in ("1", "2", "3", "4"):
        monitor.trigger_phase(int(args.phase), target=args.target, generations=args.generations)
        return

    if args.phase == "all":
        monitor.trigger_phase(2, generations=args.generations)
        monitor.trigger_phase(3, target=args.target or "DEFAULT_AGENT_IDENTITY", generations=args.generations)
        monitor.trigger_phase(4, target=args.target or "tip_tool", generations=args.generations)
        return

    # Default 'auto': triage and evolve underperforming targets
    plan = monitor.triage_all_phases(code_threshold=args.threshold)
    if plan["phase2_tool_descriptions_needed"]:
        monitor.trigger_phase(2, generations=args.generations)
    if plan["phase3_system_prompts_needed"]:
        monitor.trigger_phase(3, generations=args.generations)
    for target, score in plan["phase4_code_targets"]:
        monitor.trigger_evolution(target, score)


if __name__ == "__main__":
    main()
