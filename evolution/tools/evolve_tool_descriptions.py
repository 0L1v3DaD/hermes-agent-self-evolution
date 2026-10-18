"""
HEOS Phase 2: Tool Description Evolution Loop
Uses DSPy + GEPA + Vertex AI gemini-embedding-2 semantic parameter scoring to optimize
natural language tool schemas for high-accuracy selection and zero cross-tool interference.
"""
import os
import sys
import re
import ast
import json
import argparse
from pathlib import Path
from rich.console import Console

import dspy

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.fitness import compute_embedding_similarity
from evolution.core.toolset_discovery import format_toolsets_for_prompt
from evolution.tools.tool_module import ToolSelectionModule

console = Console()


class TaskToToolDatasetBuilder:
    """Safely builds synthetic and cross-tool disambiguation datasets for the tool selection optimizer."""
    def build_synthetic_dataset(self) -> tuple[list[dspy.Example], list[dspy.Example]]:
        train_examples = [
            dspy.Example(
                task_input="Write the optimized code to output/result.json",
                selected_tool="file_tools",
                parameter_values='{"path": "output/result.json"}'
            ).with_inputs("task_input"),
            dspy.Example(
                task_input="Check the file size of evolution/code/alpha_loop.py",
                selected_tool="file_tools",
                parameter_values='{"path": "evolution/code/alpha_loop.py"}'
            ).with_inputs("task_input"),
            dspy.Example(
                task_input="Run Pytest over the tool evolution tests",
                selected_tool="terminal_tool",
                parameter_values='{"command": "pytest tests/tools/"}'
            ).with_inputs("task_input"),
            dspy.Example(
                task_input="Reverse engineer the packaged JavaScript application in /opt/app/dist/bundle using REA",
                selected_tool="mcp:rea",
                parameter_values='{"target": "/opt/app/dist/bundle"}'
            ).with_inputs("task_input"),
        ]

        dev_examples = [
            dspy.Example(
                task_input="Read the contents of evolution/tools/tool_module.py",
                selected_tool="file_tools",
                parameter_values='{"path": "evolution/tools/tool_module.py"}'
            ).with_inputs("task_input"),
            dspy.Example(
                task_input="Examine the Docker container sandboxes list",
                selected_tool="terminal_tool",
                parameter_values='{"command": "docker ps -a"}'
            ).with_inputs("task_input"),
            dspy.Example(
                task_input="Inspect the compiled ELF binary /usr/bin/target_bin for symbols and call graphs with REA",
                selected_tool="mcp:rea",
                parameter_values='{"target": "/usr/bin/target_bin"}'
            ).with_inputs("task_input"),
        ]
        return train_examples, dev_examples


_TOOL_METRIC_CACHE: dict[tuple[str, str, str, str], tuple[float, str]] = {}


def tool_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """DSPy-compatible metric that scores tool selection accuracy and JSON parameter correctness.
    Uses exact match + Vertex AI gemini-embedding-2 semantic similarity for parameter values,
    and caches evaluations across GEPA's rollout and reflection passes.
    """
    pred_tool = (getattr(prediction, "selected_tool", "") or "").strip()
    gold_tool = (getattr(example, "selected_tool", "") or "").strip()
    pred_params_str = (getattr(prediction, "parameter_values", "") or "").strip()
    gold_params_str = (getattr(example, "parameter_values", "") or "").strip()

    cache_key = (gold_tool.lower(), pred_tool.lower(), gold_params_str, pred_params_str)
    if cache_key in _TOOL_METRIC_CACHE:
        cached_score, cached_feedback = _TOOL_METRIC_CACHE[cache_key]
        return dspy.Prediction(score=cached_score, feedback=cached_feedback)

    # 1. Selection score
    if pred_tool.lower() != gold_tool.lower():
        res = (0.0, f"Incorrect tool selected. Expected: {gold_tool}, Got: {pred_tool}")
        _TOOL_METRIC_CACHE[cache_key] = res
        return dspy.Prediction(score=res[0], feedback=res[1])

    # 2. JSON parameter validation score
    if not pred_params_str:
        res = (0.0, "Parameter values field is empty!")
        _TOOL_METRIC_CACHE[cache_key] = res
        return dspy.Prediction(score=res[0], feedback=res[1])

    try:
        pred_params = json.loads(pred_params_str)
        gold_params = json.loads(gold_params_str)
        if not isinstance(pred_params, dict) or not isinstance(gold_params, dict):
            res = (0.0, "Parameter values must be a JSON object/dictionary.")
            _TOOL_METRIC_CACHE[cache_key] = res
            return dspy.Prediction(score=res[0], feedback=res[1])

        total_keys = len(gold_params)
        if total_keys == 0:
            res = (1.0, "Matches perfectly.")
            _TOOL_METRIC_CACHE[cache_key] = res
            return dspy.Prediction(score=res[0], feedback=res[1])

        key_scores = 0.0
        missing_or_mismatched: list[str] = []
        for k, v in gold_params.items():
            if k not in pred_params:
                missing_or_mismatched.append(f"missing key '{k}' (expected '{v}')")
                continue
            pred_v_str = str(pred_params[k]).strip()
            gold_v_str = str(v).strip()
            if pred_v_str == gold_v_str:
                key_scores += 1.0
            else:
                # Check semantic equivalence via gemini-embedding-2 for command/query strings
                emb_sim = compute_embedding_similarity(pred_v_str, gold_v_str)
                if emb_sim is not None and emb_sim >= 0.80:
                    key_scores += round(0.75 * emb_sim, 4)
                    missing_or_mismatched.append(
                        f"key '{k}' semantically close (sim={emb_sim:.2f}: got '{pred_v_str}', expected '{gold_v_str}')"
                    )
                else:
                    missing_or_mismatched.append(
                        f"key '{k}' mismatch (got '{pred_v_str}', expected '{gold_v_str}')"
                    )

        score = round(key_scores / total_keys, 4)
        detail = "; ".join(missing_or_mismatched) if missing_or_mismatched else "All parameters matched exactly."
        res = (score, f"Correct tool ({gold_tool}). Parameter precision: {score*100:.1f}%. {detail}")
        _TOOL_METRIC_CACHE[cache_key] = res
        return dspy.Prediction(score=res[0], feedback=res[1])
    except Exception as e:
        res = (0.0, f"Malformed JSON parameter values: {e}")
        _TOOL_METRIC_CACHE[cache_key] = res
        return dspy.Prediction(score=res[0], feedback=res[1])


class ToolDescriptionOptimizer:
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.hermes_path = get_hermes_agent_path()
        self.repo_root = Path(__file__).resolve().parents[2]

        master_key = config.api_key
        if not master_key:
            raise ValueError("[SRE FAIL-CLOSED] LITELLM_MASTER_KEY is not set!")

        self.lm = dspy.LM(
            model=config.optimizer_model,
            api_key=master_key,
            api_base=config.api_base,
            max_tokens=32768,
        )
        self.eval_lm = dspy.LM(
            model=config.eval_model,
            api_key=master_key,
            api_base=config.api_base,
            max_tokens=16384,
        )

    def load_original_descriptions(self) -> str:
        """Reads initial raw instructions from the local tool_module.py signature file safely."""
        tool_module_path = Path(__file__).resolve().parent / "tool_module.py"
        if not tool_module_path.exists():
            # Fallback to hermes_path if running inside an external checkout
            alt_path = self.hermes_path / "evolution" / "tools" / "tool_module.py"
            if alt_path.exists():
                tool_module_path = alt_path
            else:
                raise FileNotFoundError(f"[SRE FAIL-CLOSED] tool_module.py not found at: {tool_module_path}")

        source_code = tool_module_path.read_text()
        tree = ast.parse(source_code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "TaskToTool":
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring.strip()

        return "Given a high-level user task instruction, select the best tool from the whitelisted set."

    def evolve_descriptions(self, generations: int = 2) -> str:
        desc_str = self.load_original_descriptions()
        console.print(f"[*] Loaded baseline tool descriptions:\n{desc_str}")

        builder = TaskToToolDatasetBuilder()
        trainset, devset = builder.build_synthetic_dataset()
        max_calls = max(8, generations * (len(trainset) + len(devset)))

        with dspy.context(lm=self.eval_lm):
            module = ToolSelectionModule(desc_str)

            console.print(f"[*] Running DSPy GEPA optimization loop (max_metric_calls={max_calls})...")
            optimizer = dspy.GEPA(
                metric=tool_fitness_metric,
                reflection_lm=self.lm,
                max_metric_calls=max_calls,
                num_threads=1,
                warn_on_score_mismatch=False,
            )  # type: ignore

            optimized_module = optimizer.compile(
                student=module,
                trainset=trainset,
                valset=devset,
            )

            evolved_descriptions = optimized_module.predictor.predict.signature.instructions  # type: ignore

            if not evolved_descriptions or evolved_descriptions.strip() == desc_str.strip():
                console.print("[green]✓ GEPA Converged: Tool descriptions are already optimal. No mutation applied.[/green]")
                optimized_module.update_descriptions(desc_str)
                return desc_str

            console.print(f"[green]✓ GEPA Optimization Successful! Evolved Descriptions:[/green]\n{evolved_descriptions}")
            optimized_module.update_descriptions(evolved_descriptions)
            return evolved_descriptions


def main():
    parser = argparse.ArgumentParser(description="HEOS Phase 2: Evolve Tool Descriptions")
    parser.add_argument("--generations", type=int, default=2, help="Number of GEPA generations")
    args = parser.parse_args()

    config = EvolutionConfig()
    optimizer = ToolDescriptionOptimizer(config)
    optimizer.evolve_descriptions(generations=args.generations)


if __name__ == "__main__":
    main()
