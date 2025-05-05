import ray
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ray.init()

# Define a Ray Serve deployment
@serve.deployment(num_replicas=1)
class ModelServer:
    def __init__(self):
        model_name = "gpt2" # using a well-known model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    async def __call__(self, request):
        prompt = request.query_params.get("prompt", "Tell me about Ray")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=2048,  # Adjust to your requirement
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}

app = ModelServer.bind()

