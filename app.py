# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:56:02 2025

@author: jveraz
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

class QueryInput(BaseModel):
    inputs: str

@app.post("/")
async def generate(query: QueryInput):
    input_ids = tokenizer(query.inputs, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=200)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated}
