from typing import Dict, Type

# Registry for LLM API classes
tllm_registry: Dict[str, Type] = {}


def register_llm(name: str):
    """
    Decorator to register a new LLM API class under the given name.
    Usage:
        @register_llm('mycustom')
        class MyCustomAPI(BaseLLMAPI):
            ...
    Then get_llm_api('mycustom', ...) will instantiate MyCustomAPI.
    """
    def decorator(cls: Type):
        key = name.lower()
        tllm_registry[key] = cls
        return cls
    return decorator


def get_llm_api(name: str, **kwargs):
    """
    Factory function to instantiate an LLM API by name.

    Args:
        name: name of the registered LLM (case-insensitive).
        kwargs: arguments to pass to the constructor of the API class.

    Raises:
        ValueError: if the name is not registered.
    """
    key = name.lower()
    if key not in tllm_registry:
        raise ValueError(f"Unsupported LLM type: {name}. Available: {list(tllm_registry.keys())}")
    api_cls = tllm_registry[key]
    return api_cls(**kwargs)

# Example registrations:
from llm_apis.openai_api import OpenAIAPI
from llm_apis.claude_api import ClaudeAPI
from llm_apis.llama2_api import Llama2API
from llm_apis.llama3_api import Llama3API

register_llm('openai')(OpenAIAPI)
register_llm('claude')(ClaudeAPI)
register_llm('llama2')(Llama2API)
register_llm('llama3')(Llama3API)

# Now users can define and register custom APIs:
# @register_llm('myapi')
# class MyCustomAPI(BaseLLMAPI):
#     ...
