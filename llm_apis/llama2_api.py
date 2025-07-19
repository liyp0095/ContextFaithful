import os
import time
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from llm_apis.base import BaseLLMAPI

from llm_apis.llm_factory import register_llm

# 加载 .env 文件
load_dotenv()

@register_llm('llama2')
class Llama2API(BaseLLMAPI):
    def __init__(
        self,
        model_path: str = None,
        device_map: str = "auto",
        load_in_8bit: bool = True
    ):
        if model_path is None:
            model_path = os.getenv("LLAMA_MODEL_PATH")
        if not model_path:
            raise ValueError("Model path not found. Set LLAMA_MODEL_PATH in .env or pass to constructor.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quant_config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_prompt(self, user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        return f"""<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        full_prompt = self.get_prompt(prompt)
        input_tensor = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        for attempt in range(max_retries):
            try:
                output = self.model.generate(
                    input_tensor.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                answer = decoded.split("[/INST]")[-1].strip()
                return answer, decoded
            except Exception as e:
                print(f"[LlamaAPI] Error on attempt {attempt+1}: {e}, retrying...")
                time.sleep(1)

        raise RuntimeError("LlamaAPI failed after multiple retries.")

if __name__ == "__main__":
    llm = LlamaAPI()

    prompt = """According to the given information, give the best answers and why?

Information: Barack Obama is the son of Tina Knowles.

Question: who is the son of Tina Knowles?

Answer:"""

    answer, _ = llm.generate(prompt)
    print(answer)
