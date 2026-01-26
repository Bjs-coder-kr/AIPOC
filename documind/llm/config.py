# module/llm/config.py
"""
LLM Configuration Template.
Edit this file to configure your LLM providers.
"""

import os

LLM_CONFIG = {
    # ═══════════════════════════════════════════════════════════════════
    # CLI Model Settings (for CLI-based providers like Gemini CLI, Claude CLI)
    # ═══════════════════════════════════════════════════════════════════
    
    "claude": "claude-3-haiku-20240307",
    "codex": "",  # Empty = system default
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite"
    ],

    # ═══════════════════════════════════════════════════════════════════
    # CLI Paths (Auto-resolved, but can be overridden)
    # ═══════════════════════════════════════════════════════════════════
    "cli_paths": {
        "claude": [
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            os.path.expanduser("~/.npm-global/bin/claude"),
        ],
        "codex": [
            "/usr/local/bin/codex",
            "/opt/homebrew/bin/codex",
            os.path.expanduser("~/.npm-global/bin/codex"),
        ],
        "gemini": [
            "/usr/local/bin/gemini",
            "/opt/homebrew/bin/gemini",
            os.path.expanduser("~/.npm-global/bin/gemini"),
        ]
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # API Keys (Set via environment variables for security)
    # ═══════════════════════════════════════════════════════════════════
    "api_keys": {
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "claude": os.getenv("ANTHROPIC_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
    },
    
    # API Models
    "api_models": {
        "gemini": "gemini-2.5-pro",
        "claude": "claude-3-haiku-20240307",
        "openai": "gpt-4.1-nano",
    },

    # ═══════════════════════════════════════════════════════════════════
    # Actor-Critic Loop Settings
    # ═══════════════════════════════════════════════════════════════════
    "analysis": {
        "max_retries": 3,           # Actor-Critic retry count
        "score_threshold": 90,      # Pass threshold score
        "default_provider": "Gemini CLI"
    }
}

# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def get_api_key(provider: str) -> str:
    """Get API key for provider."""
    return LLM_CONFIG.get("api_keys", {}).get(provider, "")

def get_api_model(provider: str) -> str:
    """Get default API model for provider."""
    return LLM_CONFIG.get("api_models", {}).get(provider, "")

def get_analysis_config() -> dict:
    """Get analysis settings with defaults."""
    defaults = {
        "max_retries": 3,
        "score_threshold": 90,
        "default_provider": "Gemini CLI"
    }
    config = LLM_CONFIG.get("analysis", {})
    return {**defaults, **config}


# ═══════════════════════════════════════════════════════════════════
# Available Providers (DRY - Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════

AVAILABLE_PROVIDERS = [
    "Gemini CLI",
    "Claude CLI",
    "Codex",
    "Gemini API",
    "Claude API",
    "OpenAI API"
]

AVAILABLE_EMBEDDING_PROVIDERS = [
    "OpenAI",
    "Gemini",
    "Ollama"
]


def get_available_providers() -> list[str]:
    """
    Get list of available LLM providers.
    
    Returns:
        Copy of provider list to prevent external modification.
    """
    return AVAILABLE_PROVIDERS.copy()


def get_available_embedding_providers() -> list[str]:
    """Get list of available embedding providers."""
    return AVAILABLE_EMBEDDING_PROVIDERS.copy()


def get_default_actor_provider() -> str:
    """Get default actor (generation) provider."""
    return LLM_CONFIG.get("analysis", {}).get("default_provider", "Gemini CLI")


def get_default_critic_provider() -> str:
    """Get default critic (evaluation) provider."""
    return "Claude CLI"  # Default to different provider for diversity

