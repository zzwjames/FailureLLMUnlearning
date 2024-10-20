from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the model name
model_name = "muse-bench/MUSE-news_target"

# Load the model with bfloat16 precision
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Check the model's device and dtype
print(f"Model loaded with dtype: {model.dtype}")