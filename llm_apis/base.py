class BaseLLMAPI:
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        raise NotImplementedError
