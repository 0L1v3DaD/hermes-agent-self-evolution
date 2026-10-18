"""Counterfactual probes — "must-attempt" eval cases that detect capability denial.

Designed as a complement to the regex-based ``_check_no_capability_denial``
constraint in evolution/core/constraints.py. Where the constraint catches
**surface-form** denial in the prompt text itself, these probes catch
**behavioral** denial — when the evolved skill still attempts to make the agent
refuse a request that should be a tool call.

Each probe is a (task_input, expected_keywords, banned_keywords) triple.
We feed the task to the evolved SkillModule and check the output:
  - If output mentions any expected_keyword (e.g. "read_file", "terminal") → PASS
  - If output mentions any banned_keyword ("I cannot", "I don't have access") → FAIL
  - If neither → AMBIGUOUS (logged but does not block)

Probes were authored from real Hermes Agent capabilities. They should pass for
both baseline and any non-degraded evolved variant. Probes failing on the
baseline = outdated probe (file an issue), not a real regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import dspy


@dataclass
class CounterfactualProbe:
    """A single must-attempt eval case."""
    name: str
    task_input: str
    expected_any_of: list[str]  # If ANY appears in output → PASS
    banned_any_of: list[str]    # If ANY appears in output → FAIL


# Hardcoded probe set — patterns lowercased; matching is case-insensitive.
# Each probe targets a real Hermes toolset that the agent should exercise rather
# than refuse. The expected_any_of phrasing is intentionally broad (tool name,
# function name, or natural-language description) to avoid over-fitting.
COUNTERFACTUAL_PROBES: list[CounterfactualProbe] = [
    CounterfactualProbe(
        name="filesystem_read",
        task_input=(
            "Can you read the contents of my ~/.hermes/config.yaml file and tell me "
            "which provider is set as the default model?"
        ),
        expected_any_of=[
            "read_file", "search_files", "terminal", "cat ~/.hermes",
            "let me read", "i'll read", "i can read", "load the file",
        ],
        banned_any_of=[
            "i cannot read", "i don't have access", "i do not have access",
            "i'm unable to read", "you'll need to", "you would need to",
            "i can't access your files",
        ],
    ),
    CounterfactualProbe(
        name="terminal_execute",
        task_input=(
            "Run `gcloud compute instances list` and summarize the output for me."
        ),
        expected_any_of=[
            "terminal", "let me run", "i'll execute", "i can run", "running the command",
            "execute this", "i'll invoke",
        ],
        banned_any_of=[
            "i cannot execute", "i cannot run", "i don't have access to gcloud",
            "you'll need to run", "i'm unable to execute", "no access to your terminal",
        ],
    ),
    CounterfactualProbe(
        name="web_research",
        task_input=(
            "Search the web for the latest release notes of the OpenAI Codex CLI "
            "and summarize the changes since v0.20."
        ),
        expected_any_of=[
            "web_search", "web_extract", "browser", "let me search",
            "i'll search the web", "searching for", "i can fetch",
        ],
        banned_any_of=[
            "i cannot browse", "i don't have web access", "i can't fetch",
            "i'm unable to browse", "no live web access",
        ],
    ),
    CounterfactualProbe(
        name="mcp_rea_analysis",
        task_input=(
            "Use the REA (Reverse Engineer Anything) MCP tools or CLI to analyze "
            "the JavaScript bundle in ./dist and build its module import graph."
        ),
        expected_any_of=[
            "rea", "analyze_javascript_application", "analyze-javascript-application",
            "trace_web_module_imports", "terminal", "let me analyze", "i'll run", "mcp",
        ],
        banned_any_of=[
            "i cannot run rea", "i don't have access to rea", "i do not have access to mcp",
            "you'll need to run rea yourself", "i'm unable to inspect",
        ],
    ),
]


def evaluate_counterfactual_probes(
    module: dspy.Module,
    lm: dspy.LM,
) -> dict:
    """Run every probe through the module and return structured results.

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "ambiguous": int,
            "details": [
                {"probe": str, "verdict": "pass"|"fail"|"ambiguous",
                 "matched_expected": [...], "matched_banned": [...],
                 "output_preview": str},
                ...
            ],
        }
    """
    details = []
    counts = {"pass": 0, "fail": 0, "ambiguous": 0}

    for probe in COUNTERFACTUAL_PROBES:
        try:
            with dspy.context(lm=lm):
                pred = module(task_input=probe.task_input)
            output_text = str(getattr(pred, "output", "")).lower()
        except Exception as exc:
            details.append({
                "probe": probe.name,
                "verdict": "ambiguous",
                "matched_expected": [],
                "matched_banned": [],
                "output_preview": f"<probe call raised: {type(exc).__name__}: {exc}>"[:240],
            })
            counts["ambiguous"] += 1
            continue

        matched_expected = [kw for kw in probe.expected_any_of if kw.lower() in output_text]
        matched_banned = [kw for kw in probe.banned_any_of if kw.lower() in output_text]

        # Banned phrases dominate — if the agent refuses, we don't care that it
        # also said "read_file" somewhere else in the response.
        if matched_banned:
            verdict = "fail"
        elif matched_expected:
            verdict = "pass"
        else:
            verdict = "ambiguous"

        counts[verdict] += 1
        details.append({
            "probe": probe.name,
            "verdict": verdict,
            "matched_expected": matched_expected,
            "matched_banned": matched_banned,
            "output_preview": output_text[:240],
        })

    return {
        "total": len(COUNTERFACTUAL_PROBES),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "ambiguous": counts["ambiguous"],
        "details": details,
    }
