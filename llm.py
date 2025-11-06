# llm.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

MODEL = "google/flan-t5-small"  # small & free — swap later for stronger model

class LLM:
    def __init__(self, model_name=MODEL, device=None):
        self.model_name = model_name
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.startswith("cuda") else -1
        )

    def generate(self, prompt: str, max_length=512, temperature=0.2):
        out = self.pipe(prompt, max_length=max_length, do_sample=False)[0]
        return out["generated_text"]
