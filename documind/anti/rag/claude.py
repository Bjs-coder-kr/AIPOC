from langchain_anthropic import ChatAnthropic

def get_claude():
    return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)


def get_claude_critic():
    return ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0)
