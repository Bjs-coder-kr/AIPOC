# module/llm/providers.py
"""
LLM Provider Abstraction Layer.
Handles CLI and API calls to various LLM providers.
"""

import shutil
import os
import subprocess
import requests
import logging

from .config import LLM_CONFIG, get_api_key, get_api_model

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CLI Path Resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_cli_path(tool_name):
    """Resolve CLI tool path from config or system PATH."""
    paths = LLM_CONFIG.get("cli_paths", {}).get(tool_name, [])
    if isinstance(paths, str): 
        paths = [paths]
    
    # 1. Check system PATH
    if shutil.which(tool_name):
        return shutil.which(tool_name)
        
    # 2. Check configured paths
    for path in paths:
        if os.path.exists(path):
            return path
            
    return f"/usr/local/bin/{tool_name}"  # Default fallback

# Cached paths
CLAUDE_PATH = resolve_cli_path("claude")
CODEX_PATH = resolve_cli_path("codex")
GEMINI_PATH = resolve_cli_path("gemini")

# ═══════════════════════════════════════════════════════════════════
# API Provider Handlers
# ═══════════════════════════════════════════════════════════════════

def _call_claude_api(prompt: str) -> str:
    """Call Claude API."""
    api_key = get_api_key("claude")
    model = get_api_model("claude")
    
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found."
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["content"][0]["text"]
    return f"Claude API Error {resp.status_code}: {resp.text}"


def _call_gemini_api(prompt: str) -> str:
    """Call Gemini API."""
    api_key = get_api_key("gemini")
    model = get_api_model("gemini")
    
    if not api_key:
        return "Error: GEMINI_API_KEY not found."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return f"Gemini API Error {resp.status_code}: {resp.text}"


def _call_openai_api(prompt: str) -> str:
    """Call OpenAI API."""
    api_key = get_api_key("openai")
    model = get_api_model("openai")
    
    if not api_key:
        return "Error: OPENAI_API_KEY not found."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"OpenAI API Error {resp.status_code}: {resp.text}"


# API Handler mapping
_API_HANDLERS = {
    "Claude API": _call_claude_api,
    "Gemini API": _call_gemini_api,
    "OpenAI API": _call_openai_api,
}


def _call_api_provider(provider: str, prompt: str) -> str:
    """Route to appropriate API handler."""
    try:
        for key, handler in _API_HANDLERS.items():
            if key in provider:
                return handler(prompt)
        return "Error: Unknown API Provider"
    except Exception as e:
        return f"API Exception: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# Main LLM Call Function
# ═══════════════════════════════════════════════════════════════════

def call_llm(provider: str, prompt: str) -> str:
    """
    Call LLM via CLI or API based on provider string.
    
    Supported providers:
    - "Claude CLI", "Gemini CLI", "Codex" (CLI-based)
    - "Claude API", "Gemini API", "OpenAI API" (API-based)
    """
    # 1. API Calls
    if "API" in provider:
        return _call_api_provider(provider, prompt)

    # 2. CLI Calls
    try:
        if "Claude CLI" in provider or "claudecli" in provider:
            model = LLM_CONFIG.get("claude", "")
            cmd = [
                CLAUDE_PATH, 
                "--print",
                "--dangerously-skip-permissions",
                "--model", model,
                prompt
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
        elif "Codex" in provider or "codexcli" in provider:
            model = LLM_CONFIG.get("codex", "")
            outfile = "codex_out.txt"
            if os.path.exists(outfile): 
                os.remove(outfile)
            
            cmd_args = [CODEX_PATH, "--full-auto"]
            if model:
                cmd_args.extend(["--model", model])
            cmd_args.extend(["exec", "-", "--output-last-message", outfile, "--skip-git-repo-check"])
            
            result = subprocess.run(cmd_args, input=prompt, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(outfile):
                with open(outfile, "r", encoding="utf-8") as f:
                    return f.read()
            
        elif "Gemini CLI" in provider or "geminicli" in provider:
            models = LLM_CONFIG.get("gemini", [])
            if isinstance(models, str): 
                models = [models]
            
            last_error = ""
            for model in models:
                try:
                    cmd = [GEMINI_PATH, "--yolo", "--model", model, prompt]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        return result.stdout.strip()
                    else:
                        error_msg = result.stderr.strip() or result.stdout.strip()
                        logger.warning(f"⚠️ Gemini Model '{model}' Failed: {error_msg[:100]}. Trying next...")
                        last_error = error_msg
                        continue
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Gemini Model '{model}' Timeout. Trying next...")
                    last_error = "Timeout"
                    continue
                except Exception as e:
                    last_error = str(e)
                    continue
            
            return f"Error: All Gemini models failed. Last error: {last_error}"
            
        else:
            return "Error: Unknown Provider"

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error ({result.returncode}): {result.stderr}"

    except Exception as e:
        return f"Exception calling LLM: {str(e)}"

# Alias for backward compatibility
call_llm_cli = call_llm
