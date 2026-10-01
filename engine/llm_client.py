"""
Universal LLM client using LangChain's init_chat_model.
Allows switching providers (Google, OpenAI, Anthropic, etc.) via .env
"""

import os
import logging
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
logger = logging.getLogger(__name__)

_model_instance = None


def get_llm():
    """
    Returns an initialized LangChain ChatModel instance.
    Supports: groq, google_genai, openai, etc.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    model_provider = os.getenv("LLM_PROVIDER", "groq")

    try:
        _model_instance = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=0.1,
            max_tokens=500,
        )
        logger.info(f"Initialized LLM ({model_provider} : {model_name})")
        return _model_instance
    except Exception as e:
        logger.warning(f"Could not initialize LLM client: {e}")
        return None
