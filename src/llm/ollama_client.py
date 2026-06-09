from langchain_ollama import ChatOllama
from configs import settings

def create_ollama_model(name: str, temperature: float = 0.1, format: str = None, with_thinking: bool = True):
    """Create a ChatOllama model"""
    model = ChatOllama(
        model=name,
        temperature=temperature,
        base_url=settings.OLLAMA_HOST,
        format=format,
        num_ctx=4096,
        reasoning=with_thinking
    )
    return model
