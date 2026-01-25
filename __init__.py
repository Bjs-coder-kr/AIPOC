"""DocuMind unified package (optimization + analysis + RAG)."""

from .target_optimizer import TargetOptimizer, generate_target_rewrite, TargetPersona, get_persona
from .actor_critic import generate_with_critic_loop, call_critic
from .llm import call_llm, LLM_CONFIG

__version__ = "1.0.0"

__all__ = [
    # Target Optimizer
    "TargetOptimizer",
    "generate_target_rewrite",
    "TargetPersona",
    "get_persona",
    # Actor-Critic
    "generate_with_critic_loop",
    "call_critic",
    # LLM
    "call_llm",
    "LLM_CONFIG",
]
