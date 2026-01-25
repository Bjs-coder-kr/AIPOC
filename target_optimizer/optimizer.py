# module/target_optimizer/optimizer.py
"""
Target Optimizer Module - Portable Version.
Adapts document complexity for different audience personas.

Features:
- Persona-based document rewriting
- Actor-Critic quality loop
- Safety guardrails for hallucination prevention
"""

import json
import re
import logging

from ..llm import call_llm
from ..actor_critic import generate_with_critic_loop
from .personas import get_persona, STRING_TO_ENUM, PERSONA_GUIDES, TargetPersona
from .guardrail import TargetGuardrail

logger = logging.getLogger(__name__)


class TargetOptimizer:
    """
    Main optimizer class for adapting text to target personas.
    
    Usage:
        optimizer = TargetOptimizer("Gemini CLI")
        result = optimizer.analyze(text, "public")
        print(result["rewritten_text"])
    """
    
    def __init__(self, provider: str):
        """
        Initialize optimizer.
        
        Args:
            provider: LLM provider string (e.g., "Gemini CLI", "OpenAI API")
        """
        self.provider = provider
        self.max_retries = 3

    def analyze(
        self, 
        text: str, 
        target_level: str = "public", 
        progress_callback=None, 
        critic_provider=None,
        chunk_size: int = 3000
    ) -> dict:
        """
        Main analysis pipeline.
        
        Args:
            text: Input text to optimize
            target_level: Target persona ("public", "student", "worker", "expert")
            progress_callback: Optional callback(stage, current, total, score, feedback, message)
            critic_provider: Optional separate LLM for critique
            chunk_size: Maximum chunk size for processing
        
        Returns:
            Dict with keys: rewritten_text, analysis, keywords, quizzes, original_text
        """
        # Simple chunking (no external dependency)
        chunks = self._split_text(text, chunk_size)
        chunk_results = []
        
        total_chunks = len(chunks)
        if progress_callback:
            progress_callback("start", 0, total_chunks, 0, "", f"🔄 Processing {total_chunks} chunk(s)...")

        for i, chunk_text in enumerate(chunks):
            chunk_progress = None
            if progress_callback:
                def make_chunk_progress(chunk_idx):
                    def inner(status, step, total, score, feedback, log):
                        if status not in ["passed", "failed"]:
                            progress_callback(status, chunk_idx+1, total_chunks, score, feedback, f"[Chunk {chunk_idx+1}/{total_chunks}] {log}")
                    return inner
                chunk_progress = make_chunk_progress(i)

            logger.info(f"🧩 Processing Chunk {i+1}/{total_chunks}...")
            res = self._process_single_chunk(chunk_text, target_level, chunk_progress, critic_provider)
            chunk_results.append(res)
            
        # Merge results
        merged = self._merge_results(chunk_results)
        return self._ensure_defaults(merged, text, merged.get("keywords", []))

    def _split_text(self, text: str, chunk_size: int) -> list:
        """Simple text chunking by character count."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current = 0
        while current < len(text):
            end = min(current + chunk_size, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_sep = text[current:end].rfind(sep)
                    if last_sep > chunk_size * 0.5:
                        end = current + last_sep + len(sep)
                        break
            chunks.append(text[current:end])
            current = end
        return chunks

    def _merge_results(self, results: list) -> dict:
        """Merge chunk results."""
        if len(results) == 1:
            return results[0]
        
        merged_text = "\n\n".join(r.get("rewritten_text", "") for r in results)
        all_keywords = []
        for r in results:
            all_keywords.extend(r.get("keywords", []))
        
        return {
            "rewritten_text": merged_text,
            "analysis": {"score": 5, "comment": f"Merged from {len(results)} chunks"},
            "keywords": list(set(all_keywords)),
            "quizzes": []
        }

    def _process_single_chunk(self, text: str, target_level: str, progress_callback, critic_provider=None) -> dict:
        """Process a single chunk with persona-based rewriting."""
        persona_enum = STRING_TO_ENUM.get(target_level.lower(), TargetPersona.PUBLIC)
        guide = PERSONA_GUIDES[persona_enum]
        guardrail = TargetGuardrail()
        
        complexity_score = self._calculate_complexity(text)
        strategy = self._route_strategy(complexity_score, persona_enum)
        
        logger.info(f"📊 [Router] Level: {target_level}, Score: {complexity_score:.2f}, Strategy: {strategy}")
        
        if progress_callback:
            progress_callback("analyzing", 0, 3, 0, "", f"Strategy: {strategy} (Complexity: {complexity_score:.2f})")

        try:
            if strategy == "PLAN_AND_SOLVE":
                result = self._execute_plan_and_solve(text, guide, complexity_score, progress_callback, critic_provider)
            else:
                result = self._execute_direct_rewrite(text, guide, progress_callback, critic_provider)
            
            rewritten_text = result.get("rewritten_text", "")
            if not guardrail.verify_all(text, rewritten_text):
                logger.warning("🛡️ [Guardrail] Verification Failed. Executing Fallback...")
                result["analysis"]["comment"] += " | ⚠️ Guardrail Failed -> Fallback"
                return self._fallback_routine(text, [])
            
            logger.info("✅ [Guardrail] Verification Passed.")
            return result
                
        except Exception as e:
            logger.error(f"❌ Strategy Execution Failed: {e}")
            return self._fallback_routine(text, [])

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        tokens = len(text.split())
        len_score = min(1.0, tokens / 500)
        
        jargon_candidates = re.findall(r'[가-힣A-Za-z]{2,}', text)
        jargon_count = len(jargon_candidates)
        jargon_ratio = min(1.0, (jargon_count / max(tokens, 1)) * 0.5) 
        
        return (0.3 * len_score) + (0.7 * jargon_ratio)

    def _route_strategy(self, score: float, persona) -> str:
        """Determine processing strategy."""
        if persona == TargetPersona.EXPERT:
            return "DIRECT_STRATEGY"
        if score > 0.4:
            return "PLAN_AND_SOLVE"
        return "DIRECT_STRATEGY"

    def _execute_plan_and_solve(self, text: str, guide: dict, complexity: float, progress_callback, critic_provider=None) -> dict:
        """Two-stage: Planner -> Editor with RAG context."""
        from ..utils.json_utils import extract_json
        
        # Step 1: Planner
        planner_prompt = f"""
**[SYSTEM PROMPT]**
You are a content strategist. Analyze the text and create a rewriting plan for audience: "{guide['role']}".

**TARGET GUIDELINES:**
- Tone: {guide['tone']}
- Vocab: {guide['vocabulary']}

**TASK:**
Output a JSON list of rewriting actions:
1. Identify jargon to define or replace.
2. Identify long sentences to split.
3. Identify abstract concepts needing analogies.

**INPUT TEXT:**
{text[:2000]}

**OUTPUT FORMAT (JSON ONLY):**
{{
  "complexity_score": <float>,
  "actions": [
    {{"type": "define", "term": "...", "strategy": "..."}},
    {{"type": "split", "target": "...", "strategy": "..."}}
  ]
}}
"""
        logger.info("🧠 [Planner] Generating Plan...")
        plan_json_str = call_llm(self.provider, planner_prompt)
        plan_data = extract_json(plan_json_str) or {"actions": []}
        
        # Step 2: Editor
        editor_prompt = f"""
**[SYSTEM PROMPT]**
You are an expert editor acting as "{guide['role']}".
Rewrite the text adhering strictly to the plan provided.

**INPUT DATA:**
<original_text>
{text}
</original_text>

<rewrite_plan>
{json.dumps(plan_data, ensure_ascii=False)}
</rewrite_plan>

**CONSTRAINTS:**
1. Maintain strict factual accuracy. Do NOT invent numbers or names.
2. Follow the tone: {guide['tone']}.
3. Structure: {guide['structure']}
4. Output language: Korean (or original if not Korean).

**OUTPUT:**
Provide ONLY the rewritten text.
"""
        logger.info("✍️ [Editor] Rewriting with Actor-Critic Loop...")
        rewritten_text = generate_with_critic_loop(
            actor_provider=self.provider,
            prompt_template="{text}",
            context_text=editor_prompt,
            context_type="Target Rewrite",
            max_retries=self.max_retries,
            progress_callback=progress_callback,
            critic_provider=critic_provider,
            persona_guide=guide
        )
        
        return {
            "rewritten_text": rewritten_text,
            "analysis": {"score": 5, "comment": f"Plan-and-Solve (Actions: {len(plan_data.get('actions',[]))})"},
            "keywords": [],
            "quizzes": []
        }

    def _execute_direct_rewrite(self, text: str, guide: dict, progress_callback, critic_provider=None) -> dict:
        """Direct single-pass rewriting."""
        prompt = f"""
You are "{guide['role']}".
Rewrite the text following these guidelines:
- Tone: {guide['tone']}
- Vocab: {guide['vocabulary']}
- Structure: {guide['structure']}

Text:
{text}

Output ONLY the rewritten text (in original language).
"""
        logger.info("⚡ [Direct] Rewriting with Actor-Critic Loop...")
        rewritten_text = generate_with_critic_loop(
            actor_provider=self.provider,
            prompt_template="{text}",
            context_text=prompt,
            context_type="Target Rewrite",
            max_retries=self.max_retries,
            progress_callback=progress_callback,
            critic_provider=critic_provider,
            persona_guide=guide
        )
        
        return {
            "rewritten_text": rewritten_text,
            "analysis": {"score": 3, "comment": "Direct Strategy"},
            "keywords": [],
            "quizzes": []
        }

    def _ensure_defaults(self, result: dict, original_text: str, anchors: list) -> dict:
        """Ensure all required fields are present."""
        final = result.copy()
        final["original_text"] = original_text
        
        if "analysis" not in final or not isinstance(final["analysis"], dict):
            final["analysis"] = {"score": 3, "comment": "No analysis info"}
            
        if "quizzes" not in final or not isinstance(final["quizzes"], list):
            final["quizzes"] = []
            
        if "keywords" not in final:
            final["keywords"] = anchors
            
        if "rewritten_text" not in final:
            final["rewritten_text"] = "(Generation failed)"

        return final

    def _fallback_routine(self, text: str, anchors: list) -> dict:
        """Safe fallback: simple rewrite."""
        try:
            logger.info("🚀 Executing Fallback Routine...")
            prompt = f"Rewrite this text to be easier to read. Output ONLY the text.\n\n{text}"
            simple_text = call_llm(self.provider, prompt)
            
            if not simple_text or "Error" in simple_text:
                simple_text = f"Conversion failed. Please refer to the original text."
                
        except Exception as e:
            logger.error(f"⚠️ Fallback Failed: {e}")
            simple_text = "A critical error occurred during generation."
            
        return {
            "original_text": text,
            "rewritten_text": simple_text,
            "analysis": {"score": 3, "comment": "Fallback mode executed"},
            "quizzes": [],
            "keywords": anchors
        }


# Convenience function
def generate_target_rewrite(provider, text, level="public", progress_callback=None, critic_provider=None):
    """
    Convenience wrapper for TargetOptimizer.
    
    Args:
        provider: LLM provider string
        text: Text to optimize
        level: Target persona level
        progress_callback: Optional progress callback
        critic_provider: Optional separate LLM for critique
    
    Returns:
        Optimization result dict
    """
    optimizer = TargetOptimizer(provider)
    return optimizer.analyze(text, level, progress_callback, critic_provider=critic_provider)
