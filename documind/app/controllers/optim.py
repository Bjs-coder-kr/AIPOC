from __future__ import annotations

import streamlit as st

from documind.llm.config import (
    get_analysis_config,
    get_available_embedding_providers,
    get_default_embedding_provider,
)
from documind.utils.db import db_manager
from documind.target_optimizer import TargetOptimizer


def _provider_options() -> list[str]:
    return [
        "Gemini CLI",
        "Claude CLI",
        "Codex",
        "Gemini API",
        "Claude API",
        "OpenAI API",
    ]


def render() -> None:
    st.title("Target Optimization")

    analysis_config = get_analysis_config()
    default_provider = analysis_config.get("default_provider", "Gemini CLI")
    providers = _provider_options()
    if default_provider not in providers:
        providers = [default_provider] + providers

    provider = st.selectbox("Provider", providers, index=providers.index(default_provider))
    embed_options = get_available_embedding_providers()
    saved_embed = db_manager.get_setting("embedding_provider")
    default_embed = saved_embed if saved_embed else get_default_embedding_provider()
    if default_embed not in embed_options:
        embed_options = [default_embed] + embed_options
    embedding_provider = st.selectbox(
        "Embedding provider",
        embed_options,
        index=embed_options.index(default_embed),
    )
    if embedding_provider:
        db_manager.save_setting("embedding_provider", embedding_provider)
    level = st.selectbox("Target audience", ["public", "student", "worker", "expert"])
    text = st.text_area("Input text", height=260)

    if st.button("Optimize"):
        if not text.strip():
            st.warning("Please enter text to optimize.")
            return
        with st.spinner("Optimizing..."):
            optimizer = TargetOptimizer(provider, embedding_provider=embedding_provider)
            result = optimizer.analyze(
                text=text,
                target_level=level,
                embedding_provider=embedding_provider,
            )

        st.subheader("Rewritten text")
        st.write(result.get("rewritten_text", ""))

        analysis = result.get("analysis")
        if analysis:
            with st.expander("Analysis"):
                st.json(analysis)

        keywords = result.get("keywords") or []
        if keywords:
            st.caption("Keywords")
            st.write(", ".join(map(str, keywords)))
