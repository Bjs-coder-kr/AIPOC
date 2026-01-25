# Target Optimization Module - Integration Guide

A portable, self-contained module for **Target Optimization** and **Actor-Critic Quality Assurance** that can be integrated into any Python/Streamlit application.

## 📁 Module Structure

```
module/
├── __init__.py                 # Main exports
├── target_optimizer/           # Target Optimization Logic
│   ├── __init__.py
│   ├── optimizer.py            # TargetOptimizer class
│   ├── personas.py             # Persona definitions
│   └── guardrail.py            # Safety guardrails
│
├── actor_critic/               # Quality Assurance Loop
│   ├── __init__.py
│   └── orchestrator.py         # generate_with_critic_loop
│
├── llm/                        # LLM Provider Abstraction
│   ├── __init__.py
│   ├── providers.py            # CLI & API handlers
│   └── config.py               # Configuration template
│
├── utils/                      # Utilities
│   └── json_utils.py
│
└── INTEGRATION_GUIDE.md        # This file
```

---

## 🚀 Quick Start

### 1. Copy the Module

Copy the entire `module/` folder to your project root:

```bash
cp -r /path/to/poc/module /your/project/
```

### 2. Install Dependencies

```bash
pip install requests
```

### 3. Configure LLM Providers

Edit `module/llm/config.py`:

```python
# Set your API keys (or use environment variables)
"api_keys": {
    "gemini": os.getenv("GEMINI_API_KEY", "your-key-here"),
    "claude": os.getenv("ANTHROPIC_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
},
```

### 4. Import and Use

```python
from module import TargetOptimizer, generate_target_rewrite

# Method 1: Using the class
optimizer = TargetOptimizer("Gemini CLI")  # or "OpenAI API", etc.
result = optimizer.analyze(
    text="Your complex text here...",
    target_level="public"  # "student", "worker", "expert"
)
print(result["rewritten_text"])

# Method 2: Using the convenience function
result = generate_target_rewrite(
    provider="Gemini CLI",
    text="Your text",
    level="student"
)
```

---

## 🎯 Supported Personas

| Level | Role | Tone |
|-------|------|------|
| `public` | Science Communicator | Friendly, emoji-friendly |
| `student` | University Professor | Educational, structured |
| `worker` | Senior Consultant | Professional, concise |
| `expert` | Peer Researcher | Academic, rigorous |

---

## 🔄 Actor-Critic Loop

The module includes a quality assurance loop:

```python
from module.actor_critic import generate_with_critic_loop

result = generate_with_critic_loop(
    actor_provider="Gemini CLI",
    prompt_template="Summarize: {text}",
    context_text="Your text here",
    context_type="Summary",
    max_retries=3,
    critic_provider="Gemini CLI",  # Optional: different model for critique
    persona_guide={"tone": "Professional"}  # Optional
)
```

**How it works:**
1. **Actor** generates content
2. **Critic** evaluates (score 0-100)
3. If score < 90, Actor refines based on feedback
4. Best attempt is returned

---

## 🛡️ Safety Guardrails

The module includes automatic guardrails:
- **Grounding Check**: Prevents number/date hallucination
- **NER Guard**: Warns about unknown proper nouns

---

## 📊 Progress Tracking

For UI integration (Streamlit, etc.), use the progress callback:

```python
def my_progress(stage, current, total, score, feedback, message):
    print(f"[{stage}] {current}/{total}: {message}")
    if score:
        print(f"  Score: {score}, Feedback: {feedback}")

result = optimizer.analyze(text, "public", progress_callback=my_progress)
```

**Stages:** `start`, `analyzing`, `generating`, `evaluating`, `refining`, `passed`, `failed`

---

## 🔧 Customization

### Custom Personas

Edit `module/target_optimizer/personas.py`:

```python
PERSONA_GUIDES[TargetPersona.WORKER] = {
    "role": "Your Custom Role",
    "tone": "Your preferred tone",
    "vocabulary": "Vocabulary guidelines",
    "structure": "Output structure",
    "complexity_limit": 0.7,
    "critic_criteria": "Evaluation criteria"
}
```

### Custom LLM Providers

Add to `module/llm/providers.py`:

```python
def _call_my_api(prompt: str) -> str:
    # Your implementation
    return response

_API_HANDLERS["My API"] = _call_my_api
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Unknown Provider` | Check provider string: "Gemini CLI", "OpenAI API", etc. |
| `API Key not found` | Set environment variable or edit `config.py` |
| `Module not found` | Ensure `module/` is in your Python path |

---

## 📝 Example: Streamlit Integration

```python
import streamlit as st
from module import TargetOptimizer

st.title("Document Optimizer")

text = st.text_area("Enter text:")
level = st.selectbox("Target Audience", ["public", "student", "worker", "expert"])

if st.button("Optimize"):
    with st.spinner("Processing..."):
        optimizer = TargetOptimizer("Gemini CLI")
        result = optimizer.analyze(text, level)
    
    st.subheader("Optimized Text")
    st.write(result["rewritten_text"])
    
    st.subheader("Analysis")
    st.json(result["analysis"])
```

---

## 📄 License

MIT License - Free to use and modify.
