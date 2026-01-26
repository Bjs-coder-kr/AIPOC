#!/usr/bin/env python3
"""
AIPOC End-to-End Pipeline Test
"""

import sys
import os
import logging

# AIPOC root path addition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from documind.anti.vectorstore.chroma_raw import save_raw_docs, get_chroma
from documind.target_optimizer.optimizer import generate_target_rewrite
from documind.llm.config import get_api_key

# Optional: Antithesis Chain
try:
    from documind.anti.rag.claude import get_claude
    from documind.anti.rag.antithesis_chain import get_antithesis_chain
    HAS_ANTITHESIS = True
except ImportError:
    HAS_ANTITHESIS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("E2E_TEST")

PHILOSOPHY_DOC = """
# 기술 결정론에 대한 고찰
기술은 사회 발전의 가장 중요한 동인이며, 기술의 발전은 필연적으로 
사회 구조, 문화, 인간 의식을 변화시킨다.
"""

COUNTER_DOC = """
# 기술의 사회적 구성론
기술은 사회적 맥락에서 형성되며, 기술의 발전 방향과 사용 방식은
사회적 선택의 결과이다.
"""

def test_rag_indexing():
    print("\n[1] RAG Indexing Test")
    docs = [
        Document(page_content=PHILOSOPHY_DOC, metadata={"source": "philosophy.md"}),
        Document(page_content=COUNTER_DOC, metadata={"source": "counter.md"})
    ]
    db = save_raw_docs(docs)
    cnt = len(db.get()["ids"])
    print(f"   ✅ Indexed {cnt} documents.")
    return True

def test_rag_search():
    print("\n[2] RAG Search Test")
    db = get_chroma()
    query = "기술이 사회를 결정하는가?"
    results = db.similarity_search_with_score(query, k=2)
    print(f"   ✅ Found {len(results)} results.")
    for doc, score in results:
        print(f"      - {doc.metadata['source']} (Score: {score:.4f})")
    return len(results) > 0

def test_antithesis():
    print("\n[3] Antithesis Generation Test (LangChain)")
    if not HAS_ANTITHESIS:
        print("   ⚠️  Antithesis pipeline modules missing.")
        return False
        
    api_key = get_api_key("claude")
    if not api_key:
        print("   ⏭️  Skipping: ANTHROPIC_API_KEY not found.")
        return True
        
    try:
        llm = get_claude()
        db = get_chroma()
        retriever = db.as_retriever(search_kwargs={"k": 2})
        chain = get_antithesis_chain(llm, retriever)
        
        print("   ⏳ Generating Antithesis...")
        # Since we just indexed philosophy docs, let's ask about AI which is NOT there
        # but the prompt expects docs provided by retriever.
        # Let's provide a topic relevant to indexed docs.
        
        # chain input is "context" populated by retriever via Runnable.
        # But wait, `get_antithesis_chain` in AIPOC code:
        # chain = ( {"context": retriever | format_docs} | prompt | llm | parser )
        # So chain.invoke("query string")? 
        # No, look at `chain` definition:
        # It takes `retriever`. `retriever` usually takes string input.
        # So yes, `chain.invoke("query")` should work.
        
        input_query = "기술 결정론의 한계"
        res = chain.invoke(input_query)
        print(f"   ✅ Result Length: {len(res)}")
        print(f"   Sample: {res[:100]}...")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_target_optimizer():
    print("\n[4] Target Optimizer Test")
    # Use Gemini CLI if available, or OpenAI API
    # Since we are in E2E, let's try available providers.
    
    # Check if Gemini CLI is configured/mocked.
    # For safety, let's try "OpenAI API" if key exists, else "Gemini CLI" 
    # assuming user has it (based on config).
    
    provider = "Gemini CLI" # Default
    if get_api_key("openai"):
        provider = "OpenAI API"
    
    print(f"   Provider: {provider}")
    
    text = "Artificial Intelligence is transformative. It changes everything about how we work."
    try:
        res = generate_target_rewrite(provider, text, level="student")
        print(f"   ✅ Rewritten: {res.get('rewritten_text')[:100]}...")
        print(f"   Score: {res['analysis']['score']}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    print("🚀 Running AIPOC E2E Pipeline Test...")
    
    steps = [
        test_rag_indexing,
        test_rag_search,
        test_antithesis,
        test_target_optimizer
    ]
    
    passed = 0
    for step in steps:
        try:
            if step():
                passed += 1
        except Exception as e:
            print(f"   ❌ Error in {step.__name__}: {e}")
            
    print(f"\nTotal Passed: {passed}/{len(steps)}")
    return 0 if passed == len(steps) else 1

if __name__ == "__main__":
    sys.exit(main())
