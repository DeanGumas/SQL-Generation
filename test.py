from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

"""
tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForCausalLM.from_pretrained(".").cuda()
input_text = "#If I have a SQL table called people with columns 'name, date, count' generate a SQL query to get all peoples names"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

tokenizer = AutoTokenizer.from_pretrained("../deepseek-coder-1.3b-instruct")
model = AutoModelForCausalLM.from_pretrained("../deepseek-coder-1.3b-instruct", torch_dtype=torch.bfloat16).cuda()
messages=[
    { 'role': 'user', 'content': "If I have a SQL table called people with columns 'name, date, count' generate a SQL query to get all peoples names"}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))