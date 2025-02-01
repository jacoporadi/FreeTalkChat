import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat(message):
    system_prompt = "You are a helpful AI assistant. Answer concisely and informatively."
    
    formatted_prompt = f"""<s>[INST] <<SYS>>{system_prompt}<</SYS>>{message} [/INST]"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = re.sub(r"<<SYS>>.*?<</SYS>>", "", response, flags=re.DOTALL).strip()
    response = response.replace("[INST]", "").replace("[/INST]", "").strip()
    
    if message in response:
        response = response.replace(message, "").strip()

    return response

gr.Interface(fn=chat, inputs="text", outputs="text").launch()
