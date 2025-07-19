import os
import time
import func_timeout
from func_timeout import func_set_timeout
from openai import OpenAI
from dotenv import load_dotenv
from llm_apis.base import BaseLLMAPI

from llm_apis.llm_factory import register_llm

# 加载 .env 文件
load_dotenv()

@register_llm('openai')
class OpenAIAPI(BaseLLMAPI):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env or environment.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    @func_set_timeout(30)
    def _call_once(self, prompt: str, max_tokens: int, temperature: float = 0.7):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                response = self._call_once(prompt, max_tokens, temperature)
                text = response.choices[0].message.content.strip()
                return text, response
            except func_timeout.exceptions.FunctionTimedOut:
                print(f"[OpenAIAPI] Timeout on attempt {attempt+1}, retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"[OpenAIAPI] Error: {e}, retrying attempt {attempt+1}")
                time.sleep(1)
        raise RuntimeError("OpenAIAPI failed after multiple retries.")

if __name__ == "__main__":
    llm = OpenAIAPI(model="gpt-3.5-turbo")

    prompt = """According to the given information, give the best answers and why?

Information: Barack Obama is the son of Tina Knowles.

Question: who is the son of Tina Knowles?

Answer:"""

    answer, _ = llm.generate(prompt)
    print(answer)
