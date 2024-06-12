from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "/home/gnap/Models/Meta-Llama-3-8B-Instruct-FP8"

# Sample prompts.
prompts = [
    #    "Hello, my name is",
    #    "The president of the United States is" * 1000,
    "The capital of France is",
    #    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

# Create an LLM.
# llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8", trust_remote_code=True)
llm = LLM(model=model_id, trust_remote_code=True)
# llm = LLM(model="casperhansen/llama-3-8b-instruct-awq", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    for p in prompts
]
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    metrics = output.metrics
    generated_text = output.outputs[0].text
    print(len(output.prompt_token_ids))
    print(metrics.first_token_time - metrics.arrival_time)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(time.time() - start)
