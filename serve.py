# serve_gpt2.py
from ray import serve
from transformers import pipeline
import ray


@serve.deployment
class GPT2Responder:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2", device=-1)

    async def __call__(self, request):
        input_data = await request.json()
        prompt = input_data.get("prompt", "")
        result = self.generator(prompt, max_length=50)
        return result[0]["generated_text"]

# Deploy after the class is defined
app = GPT2Responder.bind()
