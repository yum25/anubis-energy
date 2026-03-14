# generate_prompts.py

# Prefill stress: long inputs, minimal output
# Forces the model to process large KV caches
long_prompts = [
    "Please write a comprehensive analysis of the following topic, "
    "covering historical background, current state, and future implications: "
    "the development of transformer-based language models. " * 5
] * 100

with open("prompts/long_prompts.txt", "w") as f:
    for p in long_prompts:
        f.write(p.strip() + "\n")

# Decode stress: short inputs, long outputs
# Forces many sequential token generation steps
short_prompts = ["Continue this story:"] * 100

with open("prompts/short_prompts.txt", "w") as f:
    for p in short_prompts:
        f.write(p.strip() + "\n")