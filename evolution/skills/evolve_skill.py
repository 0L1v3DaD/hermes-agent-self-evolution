"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path, resolve_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import (
    skill_fitness_metric,
    skill_fitness_metric_with_adversarial_judge,
    LLMJudge,
    FitnessScore,
)
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gemini-4-argon",
    eval_model: str = "openai/gemini-3.8-flash",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(
            f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}, "
            f"~/.hermes/skills, or ~/.agents/skills[/red]"
        )
        sys.exit(1)

    skill = load_skill(skill_path)
    try:
        display_path = skill_path.relative_to(config.hermes_agent_path)
    except ValueError:
        display_path = skill_path
    console.print(f"  Loaded: {display_path}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and save local output (no PRs/issues)")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["hermes", "claude-code", "copilot"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["body"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy with explicit LiteLLM endpoint and generous token budget for reasoning models
    lm = dspy.LM(
        model=eval_model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=32768,
    )
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        # DSPy 3.2+ renamed max_steps; use a reflection model + max_full_evals.
        # Use max_tokens=32768 so reasoning models (e.g. gemini-4-argon) never
        # exhaust their token budget in <thought> blocks and return NoneType.
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric_with_adversarial_judge,
            max_full_evals=iterations,
            warn_on_score_mismatch=False,
            reflection_lm=dspy.LM(
                model=config.optimizer_model,
                api_base=config.api_base,
                api_key=config.api_key,
                max_tokens=32768,
            ),
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # BUG-FIX (2026-05-10): GEPA optimizes signature.instructions on the
    # inner predictor, NOT the wrapper module's `skill_text` attribute.
    # The previous code read the original skill back as "evolved", producing
    # byte-identical output that always failed validation.
    #
    # We now extract the actual evolved instructions (the prompt prefix
    # GEPA mutated each iteration) and treat it as a "prompt enhancement"
    # that sits ABOVE the original skill body. The skill body is preserved
    # verbatim — that's the right thing because GEPA never had a chance to
    # mutate it (it was passed as an input string, not a parameter).
    try:
        evolved_instruction = optimized_module.predictor.predict.signature.instructions
    except AttributeError:
        # Fallback for MIPROv2 path or future DSPy reshuffles
        try:
            evolved_instruction = optimized_module.predictor.signature.instructions
        except AttributeError:
            evolved_instruction = ""

    baseline_instruction = SkillModule(skill["body"]).predictor.predict.signature.instructions
    evolved_prefix = evolved_instruction.strip()

    if evolved_prefix == baseline_instruction.strip():
        console.print("[yellow]⚠ GEPA did not improve the prompt — evolved == baseline. Saving baseline as is.[/yellow]")
        evolved_body = skill["body"]
    else:
        # PATCH 3 (2026-05-10) — ADDITIVE REASSEMBLY
        # Previous behavior REPLACED the skill body with just the GEPA-evolved
        # prompt prefix, throwing away all reference content (commands, pitfalls,
        # examples, links). That is destructive and produces a skill that
        # passes constraints but loses the actual operational knowledge the
        # user invested time accumulating.
        #
        # New behavior: the evolved instruction sits ABOVE the original body
        # as a "Behavioral Guidance" prefix block. The original body is
        # preserved VERBATIM. This way:
        #   - GEPA's behavioral improvements (anti-hallucination, tool grounding,
        #     counterfactual handling) are injected at the top where they steer
        #     the agent's first decisions.
        #   - All factual/procedural content (commands, file paths, pitfall
        #     catalogs) survives intact for downstream lookup.
        #   - Validators run against the merged artifact, so size_limit and
        #     similarity_floor reflect the real shipping artifact, not a
        #     stripped-down version.
        evolved_prefix = evolved_instruction.strip()
        if len(evolved_prefix) > config.max_prompt_prefix_size:
            target_len = max(1000, config.max_prompt_prefix_size - 200)
            paragraphs = evolved_prefix.split("\n\n")
            kept_paragraphs: list[str] = []
            current_len = 0
            for para in paragraphs:
                sep = 2 if kept_paragraphs else 0
                if current_len + sep + len(para) <= target_len:
                    kept_paragraphs.append(para)
                    current_len += sep + len(para)
                else:
                    break
            evolved_prefix = "\n\n".join(kept_paragraphs).strip() or evolved_prefix[:target_len].rstrip()
            try:
                sig = optimized_module.predictor.predict.signature
                optimized_module.predictor.predict.signature = sig.with_instructions(evolved_prefix)
            except Exception:
                pass
            console.print(
                f"  [yellow]Compacted over-budget GEPA prefix from {len(evolved_instruction.strip())} "
                f"to {len(evolved_prefix)} chars (limit {config.max_prompt_prefix_size})[/yellow]"
            )
        prefix_block = (
            "<!-- BEGIN GEPA-EVOLVED BEHAVIORAL GUIDANCE -->\n"
            "## Behavioral Guidance (GEPA-evolved)\n\n"
            f"{evolved_prefix}\n"
            "<!-- END GEPA-EVOLVED BEHAVIORAL GUIDANCE -->\n\n"
            "---\n\n"
        )
        evolved_body = prefix_block + skill["body"]
        console.print(f"  Evolved prefix: {len(evolved_prefix)} chars (baseline prompt: {len(baseline_instruction)} chars)")
        console.print(f"  Original body preserved: {len(skill['body'])} chars verbatim")
        console.print(f"  Final merged body: {len(evolved_body)} chars (additive)")

    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    if evolved_instruction.strip() != baseline_instruction.strip():
        prefix_constraints = validator.validate_all(
            evolved_prefix,
            "prompt_prefix",
            baseline_text=baseline_instruction,
        )
        # Tier 2: full-artifact checks (structure, scripts, safety, capability-denial)
        full_constraints = validator.validate_all(
            evolved_full,
            "skill",
            baseline_text=skill["body"],
            skill_dir=skill_path.parent,
        )
        # Merge: keep prefix's size/growth verdicts, full's everything-else verdicts
        size_growth_names = {"size_limit", "growth_limit"}
        evolved_constraints = (
            [c for c in prefix_constraints if c.constraint_name in size_growth_names]
            + [c for c in full_constraints if c.constraint_name not in size_growth_names]
        )
    else:
        # No evolution happened — validate baseline as-is for completeness
        evolved_constraints = validator.validate_all(
            evolved_full,
            "skill",
            baseline_text=skill["body"],
            skill_dir=skill_path.parent,
        )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_dir = Path("output") / skill_name / f"{timestamp}_FAILED"
        failed_dir.mkdir(parents=True, exist_ok=True)
        (failed_dir / "evolved_FAILED.md").write_text(evolved_full)
        (failed_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "skill_name": skill_name,
                    "timestamp": timestamp,
                    "constraints_passed": False,
                    "failed_constraints": [c.constraint_name for c in evolved_constraints if not c.passed],
                },
                indent=2,
            )
        )
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {failed_dir}/ and {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric_with_adversarial_judge(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_score = skill_fitness_metric_with_adversarial_judge(ex, evolved_pred)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 8b. Counterfactual capability-denial probes ─────────────────────
    # Detect behavioral capability denial that surface-form regex (Patch A)
    # might miss. Probes each ask the evolved skill to do something that
    # requires a real toolset; we flag the run if the skill teaches refusal.
    from evolution.core.counterfactual_probes import evaluate_counterfactual_probes, COUNTERFACTUAL_PROBES

    console.print(f"\n[bold]Running counterfactual capability probes ({len(COUNTERFACTUAL_PROBES)} probes)[/bold]")
    probe_results = evaluate_counterfactual_probes(optimized_module, lm)
    if probe_results["failed"] > 0:
        console.print(
            f"[red]⚠ {probe_results['failed']}/{probe_results['total']} probes detected capability denial in evolved skill[/red]"
        )
        for d in probe_results["details"]:
            if d["verdict"] == "fail":
                console.print(f"  [red]✗ {d['probe']}: matched banned={d['matched_banned']}[/red]")
                console.print(f"    output: {d['output_preview'][:160]}")
    elif probe_results["passed"] == probe_results["total"]:
        console.print(f"[green]✓ All {probe_results['total']} probes passed (no capability denial detected)[/green]")
    else:
        console.print(
            f"[yellow]~ {probe_results['passed']} pass / {probe_results['ambiguous']} ambiguous / {probe_results['failed']} fail[/yellow]"
        )

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "counterfactual_probes": {
            "total": probe_results["total"],
            "passed": probe_results["passed"],
            "failed": probe_results["failed"],
            "ambiguous": probe_results["ambiguous"],
            "details": probe_results["details"],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gemini-4-argon", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gemini-3.8-flash", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
