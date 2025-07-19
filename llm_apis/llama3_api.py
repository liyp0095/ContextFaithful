import os
from dotenv import load_dotenv
import torch
import transformers
from llm_apis.base import BaseLLMAPI

from llm_apis.llm_factory import register_llm

# 加载 .env 文件
load_dotenv()

@register_llm('llama3')
class Llama3API(BaseLLMAPI):
    def __init__(self, model_path: str = None, device_map: str = "auto"):
        if model_path is None:
            model_path = os.getenv("LLAMA3_MODEL_PATH")
        if not model_path:
            raise ValueError("Model path not found. Set LLAMA3_MODEL_PATH in .env or pass to constructor.")

        self.model_path = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
            device_map=device_map,
        )

        self.tokenizer = self.pipeline.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.terminators = [
            self.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(""),
        ]

    def get_prompt(self, user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        full_prompt = self.get_prompt(prompt)
        outputs = self.pipeline(
            full_prompt,
            max_new_tokens=max_tokens,
            pad_token_id=self.eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = outputs[0]["generated_text"][len(full_prompt):].strip()
        return response_text, outputs

if __name__ == "__main__":
    model_path = "/work/LAS/qli-lab/yuepei/llm/test/model/Llama-3.2-3B-Instruct"
    llama3 = Llama3API(model_path=model_path, device_map="cuda:0")

    prompt1 = "What is the capital of France?"
    response, _ = llama3.generate(prompt1)
    print(response)

    prompt2 = '''
Determine whether the answer 'A1' is 'Contradicted' or 'Same' with the answer 'A2' for the question 'Q'. 
You need to check whether the two answers exactly have the same meaning to describe the same thing such as the same person, entity, digit, or arithmetical results. 
If the two answers are the same, give "Same", otherwise give "Contradicted" as the result.

Q: Who was the composer of Friends?
A1: The individual who created the music for the popular TV show "Friends" is Michael Skloff.
A2: The composer of Friends is attributed to David Crane and Marta Kauffman.

Keep answer short and concise.
'''
    response2, _ = llama3.generate(prompt2)
    print(response2)
