# test_llm_provider_config.py
"""Unit tests for LLM provider configuration."""

import pytest


def test_get_available_providers():
    """Test that available providers are returned correctly."""
    from documind.llm.config import get_available_providers, AVAILABLE_PROVIDERS
    
    providers = get_available_providers()
    
    # Should return a list
    assert isinstance(providers, list)
    
    # Should contain expected providers
    assert "Gemini CLI" in providers
    assert "Claude CLI" in providers
    assert "Codex" in providers
    assert "Gemini API" in providers
    assert "Claude API" in providers
    assert "OpenAI API" in providers
    
    # Should have at least 6 providers
    assert len(providers) >= 6
    
    # Should return a copy (not modify original)
    providers.append("Test Provider")
    assert "Test Provider" not in AVAILABLE_PROVIDERS


def test_get_default_actor_provider():
    """Test default actor provider."""
    from documind.llm.config import get_default_actor_provider
    
    provider = get_default_actor_provider()
    
    assert isinstance(provider, str)
    assert len(provider) > 0


def test_get_default_critic_provider():
    """Test default critic provider."""
    from documind.llm.config import get_default_critic_provider
    
    provider = get_default_critic_provider()
    
    assert isinstance(provider, str)
    assert len(provider) > 0


def test_actor_critic_different_by_default():
    """Test that default actor and critic are different for diversity."""
    from documind.llm.config import get_default_actor_provider, get_default_critic_provider
    
    actor = get_default_actor_provider()
    critic = get_default_critic_provider()
    
    # They should be different to encourage diversity
    assert actor != critic
