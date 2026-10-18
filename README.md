# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution (HEOS Phase 1–5) uses **DSPy + GEPA** (Genetic-Pareto Prompt Evolution), **Harness² (`arXiv:2610.14224`)** `INV`/`DOM` orthogonal attribution & `SurfaceSpec` validation, **AlphaEvolve MAP-Elites**, and **Bi-Temporal Sleep-Time Memory Dreaming** (`LightMem`/`Graphiti`) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and tool code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via local LiteLLM / Vertex AI ADC calls — mutating artifacts, evaluating results against real `~/.hermes/state.db` trajectories, enforcing `SurfaceSpec` and `defusedxml`/Bandit AST safety gates, and consolidating long-term memory during offline dream cycles.

## How It Works

```
Read current skill/prompt/tool/code ──► Mine ~/.hermes/state.db & build eval dataset
                                                   │
                                                   ▼
                                       GEPA / MAP-Elites Optimizer (INV vs DOM split)
                                                   │
                                                   ▼
                                       Candidate variants ──► Evaluate + Counterfactual Probes
                                                   │
                                       Harness² SurfaceSpec + AST/Bandit + Size/Denial Gates
                                                   │
                                                   ▼
                                       Experience Attribution (confounded guard) + Dream Cycle
```

## Quick Start

```bash
# Install
git clone https://github.com/0L1v3DaD/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Phase 1: Evolve a skill (synthetic eval data or real ~/.hermes/state.db trajectories)
python -m evolution.skills.evolve_skill \
    --skill reverse-engineer-anything \
    --iterations 10 \
    --eval-source sessiondb

# Phase 2: Evolve tool descriptions (DSPy + GEPA + gemini-embedding-2)
python -m evolution.tools.evolve_tool_descriptions --generations 2

# Phase 3: Evolve system prompt sections
python -m evolution.prompts.evolve_system_prompts --section DEFAULT_AGENT_IDENTITY --generations 2

# Phase 4: Evolve tool implementation code (MAP-Elites + AST/Bandit/defusedxml Sandbox)
python -m evolution.code.alpha_loop --target tip_tool --generations 2

# Phase 5: Run continuous improvement triage & bi-temporal sleep-time dream cycle
python -m evolution.monitor.continuous_improvement --phase dream --dry-run
```

## What It Optimizes (Phases 1–5)

| Phase | Dimension (`Harness²`) | Target | Engine | Status |
|-------|------------------------|--------|--------|--------|
| **Phase 1** | `DOM` (Practitioner Knowledge) | Skill files (`SKILL.md`) | DSPy + GEPA + `gemini-embedding-2` + `SurfaceSpec` Script Validator | ✅ Implemented |
| **Phase 2** | `INV` (Harness Machinery) | Tool descriptions (`tools/*.py`, MCP/REA) | DSPy + GEPA + `gemini-embedding-2` | ✅ Implemented |
| **Phase 3** | `INV` (Harness Machinery) | System prompt sections (`prompt_builder.py`) | DSPy + GEPA + Template-Brace Guard | ✅ Implemented |
| **Phase 4** | `INV` (Harness Machinery) | Tool implementation code (`tools/*.py`) | AlphaEvolve MAP-Elites + AST/Bandit/`defusedxml` Gate | ✅ Implemented |
| **Phase 5** | `INV` + `DOM` | Continuous improvement & sleep-time memory dreaming | `ContinuousImprovementMonitor` + Bi-Temporal Fact Consolidation + `confounded` Attribution Guard | ✅ Implemented |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size & growth limits** — Skills ≤15KB, tool descriptions ≤500 chars, bounded growth
3. **Capability-denial & template-brace guards** — Rejects hallucinated refusals and raw `{{` Jinja tags in always-on prompts
4. **`Harness²` `SurfaceSpec` script verification** — Every `scripts/<name>.py` referenced in a skill must exist and pass `py_compile`
5. **`Harness²` `confounded` attribution guard** — Multi-component edits without isolated rollouts are marked `confounded` and never poison the experience library
6. **Bi-temporal sleep-time memory consolidation** — Deduplicates holographic facts by subject anchor and archives superseded claims with audit trails

## License

MIT — © 2026 Nous Research & Contributors
