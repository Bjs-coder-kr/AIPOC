from langchain_anthropic import ChatAnthropic

def get_claude():
    return ChatAnthropic(
        model="claude-3-5-haiku-latest",
        temperature=0
    )
