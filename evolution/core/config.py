"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path. Discovered lazily and non-fatally so the config
    # can be constructed even when no repo is present (e.g. unit tests, or
    # callers that pass an explicit path). Use resolve_hermes_agent_path() when
    # an explicit override should win, or get_hermes_agent_path() to require one.
    hermes_agent_path: Optional[Path] = field(default_factory=lambda: _discover_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration (LiteLLM Vertex AI ADC proxy)
    optimizer_model: str = "openai/gemini-4-argon"  # Model for GEPA reflections
    eval_model: str = "openai/gemini-3.8-flash"  # Model for fast evaluation
    judge_model: str = "openai/claude-opus-5-5"  # Model for goal/dataset judging
    embedding_model: str = "gemini-embedding-2"  # Vertex AI 768d embedding via LiteLLM
    api_base: str = "http://127.0.0.1:8000/v1"

    # Constraints
    max_skill_size: int = 15_000  # 15KB default
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_prefix_size: int = 15_000  # GEPA-evolved prompt prefix size limit (15KB)
    max_prompt_growth: float = 0.2  # 20% max growth over baseline (skills)
    max_prefix_growth: float = 50.0  # Prompt prefix growth ratio over baseline instruction

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output (Strictly local-only: never open GitHub PRs or Issues)
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = False
    create_issue: bool = False

    @property
    def api_key(self) -> Optional[str]:
        """Load API key dynamically from environment or local Hermes config."""
        for env_key in ("LITELLM_MASTER_KEY", "OPENAI_API_KEY"):
            val = os.getenv(env_key)
            if val:
                return val
        gepa_env = Path.home() / ".hermes" / "secrets" / "gepa.env"
        if gepa_env.exists():
            for line in gepa_env.read_text().splitlines():
                if line.startswith(("LITELLM_MASTER_KEY=", "OPENAI_API_KEY=")):
                    return line.split("=", 1)[1].strip()
        litellm_cfg = Path.home() / "litellm-proxy" / "config.yaml"
        if litellm_cfg.exists():
            for line in litellm_cfg.read_text(errors="replace").splitlines():
                s = line.strip()
                if s.startswith("master_key:"):
                    return s.split(":", 1)[1].strip().strip("'\"")
        return None


def _discover_hermes_agent_path() -> Optional[Path]:
    """Best-effort hermes-agent repo discovery that never raises.

    Returns the discovered path, or None when no repo can be found. Used as
    the EvolutionConfig default so construction never crashes; callers that
    truly require the repo should use get_hermes_agent_path().
    """
    try:
        return get_hermes_agent_path()
    except FileNotFoundError:
        return None


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )


def resolve_hermes_agent_path(hermes_repo: Optional[str] = None) -> Path:
    """Return the hermes-agent repo path, honoring an explicit override."""
    if hermes_repo:
        return Path(hermes_repo).expanduser()
    return get_hermes_agent_path()
