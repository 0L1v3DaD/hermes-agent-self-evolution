"""
HEOS Phase 3: System Prompt Section Evolution Loop
Uses DSPy + GEPA + PACE (Two-Timescale) to optimize modular system prompt sections
while enforcing strict prompt caching alignments and behavioral guidelines.
"""
import os
import sys
import re
import ast
import argparse
from pathlib import Path
from rich.console import Console

import dspy

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.prompts.prompt_module import PromptSectionModule
from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.fitness import skill_fitness_metric

console = Console()


class SystemPromptOptimizer:
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

    def load_original_guidance(self, section_name: str = "DEFAULT_AGENT_IDENTITY") -> str:
        """Reads initial prompt guidance from Hermes prompt_builder.py or local prompt_module.py."""
        prompt_builder_path = self.hermes_path / "agent" / "prompt_builder.py"
        if prompt_builder_path.exists():
            try:
                tree = ast.parse(prompt_builder_path.read_text())
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == section_name:
                                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    return node.value.value.strip()
            except Exception:
                pass

        prompt_module_path = Path(__file__).resolve().parent / "prompt_module.py"
        if not prompt_module_path.exists():
            alt_path = self.hermes_path / "evolution" / "prompts" / "prompt_module.py"
            if alt_path.exists():
                prompt_module_path = alt_path
            else:
                raise FileNotFoundError(f"[SRE FAIL-CLOSED] prompt_module.py not found at: {prompt_module_path}")

        source_code = prompt_module_path.read_text()
        tree = ast.parse(source_code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "PromptSectionSignature":
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring.strip()

        return "Apply SRE behavioral guidelines to structure and optimize the raw system prompt section."

    def evolve_section(self, generations: int = 2, section_name: str = "DEFAULT_AGENT_IDENTITY") -> str:
        guidance_str = self.load_original_guidance(section_name=section_name)
        console.print(f"[*] Loaded baseline prompt guidance ({section_name}):\n{guidance_str}")

        dataset_dir = self.repo_root / "datasets" / "prompts" / section_name.lower()
        if (dataset_dir / "dataset.json").exists():
            from evolution.core.dataset_builder import EvalDataset
            dataset = EvalDataset.load(dataset_dir)
            console.print(f"[*] Loaded cached prompt dataset from {dataset_dir} ({len(dataset.all_examples)} cases)")
        else:
            builder = SyntheticDatasetBuilder(self.config)
            dataset = builder.generate(guidance_str, artifact_type="prompt_section", num_cases=4)
            dataset.save(dataset_dir)
            console.print(f"[*] Saved generated prompt dataset to {dataset_dir}")
        trainset = dataset.to_dspy_examples(split="train")[:2]
        valset = (dataset.to_dspy_examples(split="val") or trainset)[:1]
        max_calls = max(4, generations * (len(trainset) + len(valset) + 1))

        with dspy.context(lm=self.eval_lm):
            module = PromptSectionModule(guidance_str)

            console.print(f"[*] Running DSPy GEPA optimization loop for Prompt Section '{section_name}' (max_metric_calls={max_calls})...")
            optimizer = dspy.GEPA(
                metric=skill_fitness_metric,
                reflection_lm=self.lm,
                max_metric_calls=max_calls,
                num_threads=1,
                warn_on_score_mismatch=False,
            )  # type: ignore

            optimized_module = optimizer.compile(
                student=module,
                trainset=trainset,
                valset=valset,
            )

            evolved_guidance = optimized_module.predictor.predict.signature.instructions  # type: ignore

            if not evolved_guidance or evolved_guidance.strip() == guidance_str.strip():
                console.print("[green]✓ GEPA Converged: System prompt guidance is already optimal. No mutation applied.[/green]")
                optimized_module.update_guidance(guidance_str, section_name=section_name)
                return guidance_str

            console.print(f"[green]✓ GEPA Optimization Successful! Evolved Prompt Guidance ({section_name}):[/green]\n{evolved_guidance}")
            optimized_module.update_guidance(evolved_guidance, section_name=section_name)
            return evolved_guidance


def main():
    parser = argparse.ArgumentParser(description="HEOS Phase 3: Evolve System Prompt Sections")
    parser.add_argument("--section", type=str, default="DEFAULT_AGENT_IDENTITY", help="Prompt section name")
    parser.add_argument("--generations", type=int, default=2, help="Number of GEPA generations")
    args = parser.parse_args()

    config = EvolutionConfig()
    optimizer = SystemPromptOptimizer(config)
    optimizer.evolve_section(generations=args.generations, section_name=args.section)


if __name__ == "__main__":
    main()
