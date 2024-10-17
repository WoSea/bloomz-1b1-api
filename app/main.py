from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import time
import torch.nn.functional as F
from threading import Thread
import requests
import logging

app = FastAPI()

#load the model and tokenizer
model_name = "bigscience/bloomz-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_fp16 = AutoModelForCausalLM.from_pretrained(model_name).half().cuda() #use half() to quantized
model_original = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
@app.post("/llm")
async def llm(prompt: str):
    tokens_per_second = []
    num_runs = 5
    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        #start time
        start_time = time.time()
        outputs = model_fp16.generate(**inputs, max_length=50, return_dict_in_generate=True, output_scores=True)
        end_time = time.time()

        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        scores = [score.detach().cpu().numpy() for score in outputs.scores]

        generated_tokens = len(tokenizer.tokenize(generated_text)) - len(tokenizer.tokenize(input_text))
        elapsed_time = end_time - start_time
        tokens_per_second.append(generated_tokens / elapsed_time)
    
    vram_allocated = torch.cuda.memory_allocated() / 1e9
    vram_reserved = torch.cuda.memory_reserved() / 1e9
    tokens_per_sec = sum(tokens_per_second) / len(tokens_per_second)

    perplexity_fp16 = compute_perplexity(model_fp16, prompt)
    perplexity_original = compute_perplexity(model_original, prompt)

    text_fp16 = generate_text(model_fp16, prompt)
    text_original = generate_text(model_original, prompt)
 
    return {"input": prompt, "foutput": generated_text, "vram_allocated": round(vram_allocated, 2),
                    "vram_reserved": round(vram_reserved, 2), "speed": tokens_per_sec, "perplexity_fp16": perplexity_fp16,
                    "perplexity_original": perplexity_original,"text_fp16": text_fp16, "text_original": text_original}

@app.get("/alive")
def alive_check():
    return {"status": "alive"}

def compute_perplexity(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def generate_text(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run():
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)

# thread = Thread(target=run)
# thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)