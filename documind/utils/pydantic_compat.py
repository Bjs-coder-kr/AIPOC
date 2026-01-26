"""Pydantic compatibility helpers."""

from __future__ import annotations


def patch_pydantic_v1_for_chromadb() -> None:
    """Force pydantic v1 BaseSettings for chromadb on pydantic v2."""
    try:
        import pydantic
    except Exception:
        return

    version = getattr(pydantic, "VERSION", "")
    try:
        major = int(str(version).split(".", 1)[0])
    except Exception:
        return

    if major < 2:
        return

    try:
        from pydantic import v1 as pydantic_v1
    except Exception:
        try:
            import pydantic.v1 as pydantic_v1  # type: ignore
        except Exception:
            return

    pydantic.BaseSettings = pydantic_v1.BaseSettings
    pydantic.validator = pydantic_v1.validator
