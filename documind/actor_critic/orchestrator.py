# module/actor_critic/orchestrator.py
"""
Actor-Critic Loop Engine.
Exportable Module for quality assurance through iterative generation and evaluation.
"""

import logging
from dataclasses import dataclass

from ..llm import call_llm, LLM_CONFIG
from ..utils.json_utils import extract_json

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# State Model
# ═══════════════════════════════════════════════════════════════════


@dataclass
class OptimizerState:
    attempt: int
    max_retries: int
    current_score: int
    current_text: str
    feedback: str
    status: str
    decision_required: bool
    best_score: int = 0
    best_text: str = ""
    best_feedback: str = ""
    message: str = ""

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
    extra_retries=None,
    min_score_for_extra=None,
    pass_threshold=None,
    check_threshold=None,
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
    
    Yields:
        OptimizerState
    """
    # Load config
    analysis_config = LLM_CONFIG.get("analysis", {})
    max_retries = max_retries or analysis_config.get("max_retries", 5)
    extra_retries = extra_retries or analysis_config.get("extra_retries", 3)
    min_score_for_extra = min_score_for_extra or analysis_config.get("min_score_for_extra", 75)
    pass_threshold = pass_threshold or analysis_config.get("pass_threshold", 90)
    check_threshold = check_threshold or analysis_config.get("check_threshold", 85)
    archive_threshold = analysis_config.get("archive_threshold", 95)

    critic_provider = critic_provider or actor_provider 
    
    current_prompt = prompt_template.replace("{text}", context_text)
    
    logger.info(f"🚀 [Start] Actor: {actor_provider}, Critic: {critic_provider}")
    if progress_callback:
        progress_callback("start", 0, max_retries, 0, "", f"Actor: {actor_provider}, Critic: {critic_provider}")
    
    def _build_refine_prompt(score, feedback):
        persona_reminder = ""
        if persona_guide:
            persona_reminder = f"\n(Required tone: {persona_guide.get('tone')})"
        return f"""
        The previous draft received a score ({score}).
        
        [Critic Feedback]
        {feedback}
        {persona_reminder}
        
        [Instructions]
        Rewrite the draft incorporating the feedback above.
        Output only the final revised version.
        
        [Original Text]
        {context_text}
        """

    history = []
    total_retries = max_retries
    extra_used = False
    bonus_used = False
    attempt = 0

    while attempt < total_retries:
        # 1. Generate (Actor)
        logger.info(f"  🎬 [{attempt+1}/{total_retries}] Actor Generating...")
        if progress_callback:
            progress_callback(
                "generating",
                attempt + 1,
                total_retries,
                0,
                "",
                f"🎬 [{attempt+1}/{total_retries}] Actor generating...",
            )
        draft = call_llm(actor_provider, current_prompt)
        
        # 2. Evaluate (Critic)
        logger.info(f"  🧐 [{attempt+1}/{total_retries}] Critic Evaluating...")
        if progress_callback:
            progress_callback(
                "evaluating",
                attempt + 1,
                total_retries,
                0,
                "",
                f"🧐 [{attempt+1}/{total_retries}] Critic evaluating...",
            )
            
        eval_result = call_critic(critic_provider, draft, context_type, prompt_factory=critic_prompt_factory, persona_guide=persona_guide)
        
        score = eval_result.get("score", 0)
        feedback = eval_result.get("feedback", "No feedback")
        
        history.append({"score": score, "draft": draft, "feedback": feedback})
        best_attempt = max(reversed(history), key=lambda x: x["score"])
        best_score = best_attempt["score"]
        
        logger.info(f"    👉 Score: {score}, Feedback: {feedback}")
        if progress_callback:
            progress_callback(
                "evaluated",
                attempt + 1,
                total_retries,
                score,
                feedback,
                f"👉 Score: {score}/100",
            )
        
        # 3. Bonus try to reach archive threshold if already passing
        if score >= pass_threshold and not bonus_used:
            bonus_used = True
            if attempt + 1 >= total_retries:
                total_retries += 1
            logger.info(
                "  🔁 Pass reached. Retesting after refinement for %s+.",
                archive_threshold,
            )
            if progress_callback:
                progress_callback(
                    "refining",
                    attempt + 1,
                    total_retries,
                    score,
                    feedback,
                    f"🔁 Retesting for {archive_threshold}+ ...",
                )
            state = OptimizerState(
                attempt=attempt + 1,
                max_retries=total_retries,
                current_score=score,
                current_text=draft,
                feedback=feedback,
                status="REFINE_FOR_95",
                decision_required=False,
                best_score=best_score,
                best_text=best_attempt["draft"],
                best_feedback=best_attempt["feedback"],
                message=f"REFINE_FOR_{archive_threshold}",
            )
            yield state
            current_prompt = _build_refine_prompt(score, feedback)
            attempt += 1
            continue

        # 4a. Rollback check: If bonus attempt was used but score dropped, return best result
        if bonus_used and score < best_score:
            logger.warning(
                "  ⚠️ Score dropped after bonus attempt (%s -> %s). Rolling back to best result.",
                best_score,
                score,
            )
            if progress_callback:
                progress_callback(
                    "rollback",
                    attempt + 1,
                    total_retries,
                    best_score,
                    best_attempt["feedback"],
                    f"⚠️ Rollback to best score ({best_score})",
                )
            rollback_state = OptimizerState(
                attempt=attempt + 1,
                max_retries=total_retries,
                current_score=best_score,
                current_text=best_attempt["draft"],
                feedback=best_attempt["feedback"],
                status="PASS",
                decision_required=False,
                best_score=best_score,
                best_text=best_attempt["draft"],
                best_feedback=best_attempt["feedback"],
                message=f"ROLLBACK_TO_BEST ({best_score})",
            )
            return rollback_state

        # 4. Status determination
        status = "FAIL"
        decision_required = False
        message = ""
        if score >= pass_threshold:
            status = "PASS"
            message = "PASS"
            logger.info(f"  ✅ Passed! ({score} >= {pass_threshold})")
            if progress_callback:
                progress_callback(
                    "passed",
                    attempt + 1,
                    total_retries,
                    score,
                    feedback,
                    f"✅ Passed! ({score} >= {pass_threshold})",
                )
        elif check_threshold <= score < pass_threshold:
            status = "WAIT_CONFIRM"
            decision_required = True
            message = f"WAIT_CONFIRM ({score})"

        # Warn if score dropping
        if attempt > 0 and score < history[attempt - 1]["score"]:
            logger.warning(
                f"    ⚠️ Warning: Score dropped ({history[attempt-1]['score']} -> {score})"
            )

        state = OptimizerState(
            attempt=attempt + 1,
            max_retries=total_retries,
            current_score=score,
            current_text=draft,
            feedback=feedback,
            status=status,
            decision_required=decision_required,
            best_score=best_score,
            best_text=best_attempt["draft"],
            best_feedback=best_attempt["feedback"],
            message=message,
        )

        decision = None
        if decision_required:
            decision = yield state
        else:
            yield state

        if status == "PASS":
            return state

        if status == "WAIT_CONFIRM":
            if decision == "accept":
                state.status = "PASS"
                state.decision_required = False
                state.message = "ACCEPTED"
                return state
            if decision != "retry":
                decision = "retry"

        attempt += 1

        # Extend retries if needed
        if attempt >= total_retries:
            if not extra_used and best_score < min_score_for_extra and extra_retries > 0:
                total_retries += extra_retries
                extra_used = True
                logger.info(
                    "  ➕ Extending retries (best=%s < %s) -> total %s",
                    best_score,
                    min_score_for_extra,
                    total_retries,
                )
            else:
                if status == "WAIT_CONFIRM" and decision == "retry":
                    state.status = "PASS"
                    state.decision_required = False
                    state.message = "ACCEPTED_NO_RETRIES"
                    return state
                break

        # 5. Refine Prompt
        logger.info(f"  🔄 Retrying... (Score {score} < {pass_threshold})")
        if progress_callback:
            progress_callback(
                "refining",
                attempt,
                total_retries,
                score,
                feedback,
                f"🔄 Retrying... ({score} < {pass_threshold})",
            )

        current_prompt = _build_refine_prompt(score, feedback)

    # Best-of-N selection: favor latest attempt on ties
    best_attempt = max(reversed(history), key=lambda x: x["score"])
    best_score = best_attempt["score"]
    best_draft = best_attempt["draft"]
    best_feedback = best_attempt["feedback"]

    msg = f"⚠️ Below threshold (Best: {best_score}/100)"
    if progress_callback:
        progress_callback(
            "failed",
            total_retries,
            total_retries,
            best_score,
            best_feedback,
            msg,
        )

    final_state = OptimizerState(
        attempt=total_retries,
        max_retries=total_retries,
        current_score=best_score,
        current_text=best_draft,
        feedback=best_feedback,
        status="FAIL",
        decision_required=False,
        best_score=best_score,
        best_text=best_draft,
        best_feedback=best_feedback,
        message=msg,
    )

    if check_threshold <= best_score < pass_threshold:
        final_state.status = "WAIT_CONFIRM"
        final_state.decision_required = True
        decision = yield final_state
        if decision == "accept":
            final_state.status = "PASS"
            final_state.decision_required = False
            final_state.message = "ACCEPTED"
        return final_state

    return final_state
