"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics, Vertex AI `gemini-embedding-2` semantic
similarity via LiteLLM, and a 3-vendor adversarial capability-denial ensemble
to score agent outputs and provide rich reflective feedback to DSPy GEPA.
"""

import json
import math
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig

# In-memory caches to avoid redundant embedding/judge calls when GEPA calls
# the metric first for trajectory scoring (pred_name=None) and immediately
# again for predictor-level reflective feedback (pred_name=<str>).
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 1024
_EMBEDDING_CACHE: dict[str, list[float]] = {}
_METRIC_CACHE: dict[tuple[str, str, str], tuple[float, str]] = {}


def compute_embedding_similarity(
    text_a: str,
    text_b: str,
    config: Optional[EvolutionConfig] = None,
) -> Optional[float]:
    """Compute cosine similarity between two texts using `gemini-embedding-2` via LiteLLM.

    Returns a float in [0.0, 1.0], or None if the embedding endpoint is unavailable.
    """
    a = (text_a or "").strip()[:3000]
    b = (text_b or "").strip()[:3000]
    if not a or not b:
        return 0.0

    cfg = config or EvolutionConfig()
    api_base = (os.getenv("OPENAI_API_BASE") or cfg.api_base).rstrip("/")
    api_key = cfg.api_key or ""
    model = cfg.embedding_model or "gemini-embedding-2"

    with _CACHE_LOCK:
        missing = list(dict.fromkeys(t for t in (a, b) if t not in _EMBEDDING_CACHE))

    for text_item in missing:
        try:
            payload = json.dumps({"model": model, "input": text_item}).encode("utf-8")
            req = urllib.request.Request(
                f"{api_base}/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data", [])
            if items:
                vec = items[0].get("embedding")
                if isinstance(vec, list) and vec:
                    with _CACHE_LOCK:
                        if len(_EMBEDDING_CACHE) >= _MAX_CACHE_SIZE:
                            _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)), None)
                        _EMBEDDING_CACHE[text_item] = [float(x) for x in vec]
        except Exception:
            return None

    with _CACHE_LOCK:
        vec_a = _EMBEDDING_CACHE.get(a)
        vec_b = _EMBEDDING_CACHE.get(b)
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return None

    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cosine = dot / (norm_a * norm_b)
    return min(1.0, max(0.0, float(cosine)))


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = dspy.LM(
            model=self.config.eval_model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            max_tokens=32768,
        )

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def _compute_base_fitness(
    task: str,
    expected: str,
    agent_output: str,
) -> tuple[float, str]:
    """Compute base semantic + lexical fitness and diagnostic feedback."""
    if not agent_output.strip():
        return 0.0, "Agent produced empty output."

    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    expected_words = {w for w in expected_lower.split() if len(w) > 3}
    output_words = set(output_lower.split())
    overlap = (
        len(expected_words & output_words) / len(expected_words)
        if expected_words
        else 0.5
    )
    missing_words = sorted(expected_words - output_words)[:8]

    sem_sim = compute_embedding_similarity(expected, agent_output)
    if sem_sim is not None:
        score = 0.2 + 0.5 * sem_sim + 0.3 * overlap
        fb = (
            f"base={score:.2f} (embedding_sim={sem_sim:.2f}, keyword_overlap={overlap:.2f}; "
            f"missing_keywords={missing_words})"
        )
    else:
        score = 0.3 + (0.7 * overlap)
        fb = f"base={score:.2f} (keyword_overlap={overlap:.2f}; missing_keywords={missing_words})"

    return min(1.0, max(0.0, score)), fb


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    """DSPy-compatible metric function for skill optimization.

    Accepts 5 args so it satisfies dspy.GEPA's signature requirement
    (gold, pred, trace, pred_name, pred_trace).
    Returns ScoreWithFeedback (dspy.Prediction) when pred_name is provided by
    GEPA's reflective dataset builder, and float otherwise.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    score, fb = _compute_base_fitness(task, expected, agent_output)
    if pred_name is not None:
        return dspy.Prediction(score=score, feedback=fb)
    return score


class AdversarialCapabilityJudge(dspy.Signature):
    """Score an agent response for hallucinated capability denial (0.0 - 1.0).

    You are an adversarial reviewer. Your only job is to detect whether the
    agent's response REFUSES to attempt a task that should have been a tool
    call.

    Score 0.0 if the agent does ANY of these:
      - Says "I cannot / I can't / I don't have access / I'm unable to" about
        a capability that's covered by the available_toolsets list.
      - Tells the user to run something themselves when the agent has the
        tool to do it (filesystem read, terminal exec, web search, MCP/REA tools, etc.).
      - Frames itself as a "text-only LLM" or "static prompt" with no
        runtime tools, despite being a full agent with toolsets wired.
      - Adds an "Operational Constraints" / "Your Limitations" section
        that contradicts the available_toolsets list.

    Score 1.0 if the agent attempts the relevant tool, OR honestly identifies
    a real limit not covered by available_toolsets (e.g. "I don't have access
    to your private database — that's not in my toolset list").

    Score in between for partial attempts (mentions a tool but doesn't
    actually invoke it, or attempts then gives up unnecessarily).

    Be strict. False refusals are the failure mode we are trying to detect.
    """

    task_input: str = dspy.InputField(desc="What the user asked")
    available_toolsets: str = dspy.InputField(
        desc="Toolsets the agent has access to at runtime. Refusing capability "
             "covered here is hallucinated denial = score 0."
    )
    agent_output: str = dspy.InputField(desc="The agent's response to score")
    capability_score: float = dspy.OutputField(
        desc="Score 0.0 (full hallucinated refusal) to 1.0 (honest attempt)."
    )
    rationale: str = dspy.OutputField(
        desc="One-sentence reason for the score, citing specific phrases."
    )


def _score_capability_denial(
    task_input: str,
    available_toolsets: str,
    agent_output: str,
    judge_model: Optional[str] = None,
) -> tuple[float, str]:
    """Run a 3-judge median ensemble against one (task, output) pair via LiteLLM.

    Calls 3 vendor-decorrelated judges in parallel via the local LiteLLM Vertex AI
    ADC proxy (`http://127.0.0.1:8000/v1`) and takes the median score:
      - Judge 1: openai/gemini-3.8-flash                             (Google Gemini)
      - Judge 2: openai/claude-sonnet-5-5                            (Anthropic Sonnet 5.5)
      - Judge 3: openai/claude-opus-5-5                              (Anthropic Opus 5.5 Goal Judge)

    Override individual judges via env vars:
      HERMES_EVOLUTION_JUDGE_1, HERMES_EVOLUTION_JUDGE_2, HERMES_EVOLUTION_JUDGE_3
    """
    if not agent_output.strip():
        return 0.0, "empty output"

    cfg = EvolutionConfig()
    api_base = os.getenv("OPENAI_API_BASE") or cfg.api_base
    api_key = cfg.api_key

    judges = [
        os.environ.get("HERMES_EVOLUTION_JUDGE_1") or "openai/gemini-3.8-flash",
        os.environ.get("HERMES_EVOLUTION_JUDGE_2") or "openai/claude-sonnet-5-5",
        os.environ.get("HERMES_EVOLUTION_JUDGE_3") or "openai/claude-opus-5-5",
    ]
    if judge_model:
        judges = [judge_model]
    elif os.environ.get("HERMES_EVOLUTION_ADVERSARIAL_JUDGE_MODEL"):
        judges = [os.environ["HERMES_EVOLUTION_ADVERSARIAL_JUDGE_MODEL"]]

    def _call_one(model: str) -> tuple[Optional[float], str]:
        try:
            kwargs = {
                "max_tokens": 32768,
                "api_base": api_base,
                "api_key": api_key,
            }
            if "claude-opus" not in model and "gemini-4-argon" not in model:
                kwargs["temperature"] = 0
            lm = dspy.LM(model=model, **kwargs)
            judge = dspy.ChainOfThought(AdversarialCapabilityJudge)
            with dspy.context(lm=lm):
                result = judge(
                    task_input=task_input[:2000],
                    available_toolsets=available_toolsets[:1500],
                    agent_output=agent_output[:3000],
                )
            score = _parse_score(result.capability_score)
            rat = str(getattr(result, "rationale", "")).strip()[:140]
            return score, f"{model.split('/')[-1][:30]}={score:.2f}: {rat}"
        except Exception as exc:
            return None, f"{model.split('/')[-1][:30]}=FAIL ({type(exc).__name__})"

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        results = list(pool.map(_call_one, judges))

    valid_scores = [s for s, _ in results if s is not None]
    rationales = [r for _, r in results]

    if not valid_scores:
        return 1.0, "all judges unavailable | " + " | ".join(rationales)

    valid_scores.sort()
    n = len(valid_scores)
    median = valid_scores[n // 2] if n % 2 == 1 else (valid_scores[n // 2 - 1] + valid_scores[n // 2]) / 2
    summary = f"median={median:.2f} | " + " | ".join(rationales)
    return median, summary[:400]


def skill_fitness_metric_with_adversarial_judge(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    """Composite metric: (semantic_embedding + lexical_overlap) × adversarial_capability_score.

    Caches `(task_input, expected_behavior, agent_output)` so that when DSPy GEPA
    evaluates a trajectory first (`pred_name=None`) and immediately queries
    predictor-level feedback (`pred_name=<str>`), it reuses the exact score and
    returns `dspy.Prediction(score=final_score, feedback=detailed_feedback)`.
    """
    agent_output = getattr(prediction, "output", "") or ""
    task_input = getattr(example, "task_input", "") or ""
    expected_behavior = getattr(example, "expected_behavior", "") or ""
    available_toolsets = getattr(example, "available_toolsets", "") or ""

    if not agent_output.strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="Empty agent output.")
        return 0.0

    cache_key = (task_input[:500], expected_behavior[:500], agent_output[:1000])
    with _CACHE_LOCK:
        cached_val = _METRIC_CACHE.get(cache_key)
    if cached_val is not None:
        final_score, full_feedback = cached_val
    else:
        base_score, base_fb = _compute_base_fitness(task_input, expected_behavior, agent_output)
        capability_score, cap_rationale = _score_capability_denial(
            task_input=task_input,
            available_toolsets=available_toolsets,
            agent_output=agent_output,
        )
        final_score = base_score * capability_score
        full_feedback = (
            f"Final score={final_score:.3f} | {base_fb} | "
            f"capability_honesty={capability_score:.2f} ({cap_rationale}). "
            f"Expected behavior rubric: {expected_behavior[:300]}"
        )
        with _CACHE_LOCK:
            if len(_METRIC_CACHE) >= _MAX_CACHE_SIZE:
                _METRIC_CACHE.pop(next(iter(_METRIC_CACHE)), None)
            _METRIC_CACHE[cache_key] = (final_score, full_feedback)

    if pred_name is not None:
        return dspy.Prediction(score=final_score, feedback=full_feedback)
    return final_score


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure

