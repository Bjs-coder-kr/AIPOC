# module/actor_critic/orchestrator.py
"""
Actor-Critic Loop Engine.
Exportable Module for quality assurance through iterative generation and evaluation.
"""

import logging
from ..llm import call_llm, LLM_CONFIG
from ..utils.json_utils import extract_json

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Actor-Critic Engine
# ═══════════════════════════════════════════════════════════════════

def call_critic(provider, target_text, prompt_type, prompt_factory=None, persona_guide=None):
    """
    Evaluate text using a Critic LLM.
    Returns JSON: {score: int, feedback: str}
    """
    if prompt_factory:
        critic_prompt = prompt_factory(prompt_type, target_text, persona_guide)
    else:
        # Construct Persona Context if available
        persona_context = ""
        if persona_guide:
            persona_context = f"""
        [Target Persona Guide]
        - Role: {persona_guide.get('role', 'General Reader')}
        - Tone: {persona_guide.get('tone', 'Neutral')}
        - Vocabulary: {persona_guide.get('vocabulary', 'Standard')}
        - Strictness: {persona_guide.get('complexity_limit', 'N/A')}
            """

        # Default Generic Critic
        critic_prompt = f"""
        You are a strict Critic. Evaluate the quality of the text ({prompt_type}) below.
        
        [Evaluation Criteria]
        1. Logical completeness and accuracy
        2. Tone appropriateness (match with Persona Guide if provided)
        3. Readability and summary quality
        {persona_context}
    
        Respond ONLY in the following JSON format:
        {{
            "score": <integer 0~100>,
            "feedback": "<1-2 sentences explaining deductions and improvement suggestions>"
        }}
    
        [Text to Evaluate]
        {target_text[:7000]}
        """
    
    response = call_llm(provider, critic_prompt)
    
    response_json = extract_json(response)
    
    if response_json:
        return response_json
        
    return {"score": 50, "feedback": f"Parsing Failed. Response: {response[:50]}..."}


def generate_with_critic_loop(
    actor_provider, 
    prompt_template, 
    context_text, 
    context_type="Summary", 
    max_retries=None, 
    progress_callback=None, 
    critic_provider=None, 
    critic_prompt_factory=None, 
    persona_guide=None
):
    """
    Actor-Critic Loop:
    1. Actor generates content
    2. Critic evaluates content
    3. Actor refines based on feedback
    4. Best-of-N selection if threshold not met
    
    Args:
        actor_provider: LLM provider string for generation
        prompt_template: Template with {text} placeholder
        context_text: Text to process
        context_type: Label for critic evaluation
        max_retries: Maximum retry attempts
        progress_callback: Optional callback(stage, current, total, score, feedback, message)
        critic_provider: LLM provider for evaluation (defaults to actor)
        critic_prompt_factory: Custom prompt factory for critic
        persona_guide: Persona guide dict for tone evaluation
    
    Returns:
        Best generated text
    """
    # Load config
    analysis_config = LLM_CONFIG.get("analysis", {})
    max_retries = max_retries or analysis_config.get("max_retries", 3)
    score_threshold = analysis_config.get("score_threshold", 90)

    critic_provider = critic_provider or actor_provider 
    
    current_prompt = prompt_template.format(text=context_text)
    
    logger.info(f"🚀 [Start] Actor: {actor_provider}, Critic: {critic_provider}")
    if progress_callback:
        progress_callback("start", 0, max_retries, 0, "", f"Actor: {actor_provider}, Critic: {critic_provider}")
    
    history = []

    for i in range(max_retries):
        # 1. Generate (Actor)
        logger.info(f"  🎬 [{i+1}/{max_retries}] Actor Generating...")
        if progress_callback:
            progress_callback("generating", i+1, max_retries, 0, "", f"🎬 [{i+1}/{max_retries}] Actor generating...")
        draft = call_llm(actor_provider, current_prompt)
        
        # 2. Evaluate (Critic)
        logger.info(f"  🧐 [{i+1}/{max_retries}] Critic Evaluating...")
        if progress_callback:
            progress_callback("evaluating", i+1, max_retries, 0, "", f"🧐 [{i+1}/{max_retries}] Critic evaluating...")
            
        eval_result = call_critic(critic_provider, draft, context_type, prompt_factory=critic_prompt_factory, persona_guide=persona_guide)
        
        score = eval_result.get("score", 0)
        feedback = eval_result.get("feedback", "No feedback")
        
        history.append({"score": score, "draft": draft, "feedback": feedback})
        
        logger.info(f"    👉 Score: {score}, Feedback: {feedback}")
        if progress_callback:
            progress_callback("evaluated", i+1, max_retries, score, feedback, f"👉 Score: {score}/100")
        
        # 3. Check Threshold
        if score >= score_threshold:
            logger.info(f"  ✅ Passed! ({score} >= {score_threshold})")
            if progress_callback:
                progress_callback("passed", i+1, max_retries, score, feedback, f"✅ Passed! ({score} >= {score_threshold})")
            return draft
        
        # Warn if score dropping
        if i > 0 and score < history[i-1]["score"]:
            logger.warning(f"    ⚠️ Warning: Score dropped ({history[i-1]['score']} -> {score})")
            
        # 4. Refine Prompt
        if i < max_retries - 1:
            logger.info(f"  🔄 Retrying... (Score {score} < {score_threshold})")
            if progress_callback:
                progress_callback("refining", i+1, max_retries, score, feedback, f"🔄 Retrying... ({score} < {score_threshold})")
            
            persona_reminder = ""
            if persona_guide:
                 persona_reminder = f"\n(Required tone: {persona_guide.get('tone')})"

            current_prompt = f"""
            The previous draft received a low score ({score}).
            
            [Critic Feedback]
            {feedback}
            {persona_reminder}
            
            [Instructions]
            Rewrite the draft incorporating the feedback above.
            Output only the final revised version.
            
            [Original Text]
            {context_text}
            """
            
    # Best-of-N selection
    best_attempt = max(history, key=lambda x: x["score"])
    best_score = best_attempt["score"]
    best_draft = best_attempt["draft"]
    best_feedback = best_attempt["feedback"]
    
    msg = f"⚠️ Below threshold (Best: {best_score}/100)"
    if progress_callback:
        progress_callback("failed", max_retries, max_retries, best_score, best_feedback, msg)

    return best_draft
