# from llm_apis.claude_api import ClaudeAPI

# llm = ClaudeAPI(model="claude-3-5-sonnet-20240620")
# answer, _ = llm.generate("Who is the current president of France?")


# from llm_apis.openai_api import OpenAIAPI

# llm = OpenAIAPI(model="gpt-3.5-turbo")
# answer, _ = llm.generate("Who is the current president of France?")
# print(answer)  # Should print the name of the current president of France

# from llm_apis.llama2_api import Llama2API

# llm = Llama2API(model_path="/work/LAS/qli-lab/yuepei/llm/test/model/Llama-2-7b-chat-hf", load_in_8bit=False)
# answer, _ = llm.generate("Who is the current president of France?")
# print(answer)

# from llm_apis.llama3_api import Llama3API

# llm = Llama3API(model_path="/work/LAS/qli-lab/yuepei/llm/test/model/Llama-3.2-3B-Instruct", device_map="cuda:0")
# answer, _ = llm.generate("Who is the current president of France?")
# print(answer)  # Should print the name of the current president of France


# from llm_apis.llm_factory import get_llm_api

# # 之后就可以这样拿到实例
# llm = get_llm_api('claude', model='claude-3-5-sonnet-20240620')
# # llm3 = get_llm_api('llama3', model_path='/path/to/llama3')
# answer, _ = llm.generate("Who is the current president of France?")
# print(answer)  # Should print the name of the current president of France

from llm_apis.llm_factory import get_llm_api

llm = get_llm_api('llama2', model_path='/work/LAS/qli-lab/yuepei/llm/test/model/Llama-2-7b-chat-hf')
answer, _ = llm.generate("Who is the current president of France?")
print(answer)  # Should print the name of the current president of France