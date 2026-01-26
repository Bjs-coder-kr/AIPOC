from documind.anti.rag.claude import get_claude

llm = get_claude()
response = llm.invoke("한 문장으로 RAG가 뭔지 설명해줘")

print(response.content)
