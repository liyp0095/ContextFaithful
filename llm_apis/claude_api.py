import os
import time
import anthropic
import func_timeout
from func_timeout import func_set_timeout
from dotenv import load_dotenv
from llm_apis.base import BaseLLMAPI 

from llm_apis.llm_factory import register_llm

load_dotenv()

@register_llm('claude')
class ClaudeAPI(BaseLLMAPI):
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet"):
        print("api", api_key)
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in .env or environment.")
        
        print(f"[ClaudeAPI] Using model: {model}")
        print(f"[ClaudeAPI] Using API key: {api_key[:5]}... (truncated for security)")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model

    @func_set_timeout(30)
    def _call_once(self, prompt: str, max_tokens: int, temperature: float = 0.7):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                response = self._call_once(prompt, max_tokens, temperature)
                text = response.content[0].text.strip()
                return text, response
            except func_timeout.exceptions.FunctionTimedOut:
                print(f"[ClaudeAPI] Timeout on attempt {attempt+1}, retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"[ClaudeAPI] Error: {e}, retrying attempt {attempt+1}")
                time.sleep(1)
        raise RuntimeError("ClaudeAPI failed after multiple retries.")

if __name__ == "__main__":
    claude = ClaudeAPI(model="claude-3-5-sonnet-20240620")

    prompt = """According to the given information, give the best answers and why?

Information: Barack Obama is the son of Tina Knowles.

Question: who is the son of Tina Knowles?

Answer:"""

    answer, _ = claude.generate(prompt)
    print(answer)
