# FreeTalkChat

This repository contains the development of a general-purpose AI chatbot based on an open-source LLM. The chatbot is designed to provide intelligent and context-aware responses across various topics, leveraging state-of-the-art transformer models for natural language processing.

## How It Works

#### 1. **Model Selection**  
The chatbot utilizes the Mistral 7B Instruct model, an open-source LLM optimized for conversation.

#### 2. **Inference Setup**  
The model is loaded using Hugging Face's `transformers` library with support for CUDA acceleration on NVIDIA GPUs.

#### 3. **Prompt Engineering**  
User inputs are wrapped in a structured prompt format to guide the model’s response behavior.

#### 4. **Generation Parameters**  
- **Temperature**: `0.7` (balances creativity and coherence)  
- **Top-p**: `0.9` (filters unlikely words while maintaining diversity)  
- **Repetition Penalty**: `1.1` (reduces redundant outputs)  

#### 5. **Interface**  
The chatbot is accessible through a `Gradio` web interface, allowing users to interact in real-time.

## Next Steps

- Improve response quality through fine-tuning or RAG (Retrieval-Augmented Generation)  
- Deploy on a cloud-based GPU service for scalable usage  
- Implement memory for contextual conversation  

