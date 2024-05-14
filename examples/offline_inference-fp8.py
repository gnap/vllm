from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
#    "Hello, my name is",
    "The president of the United States is" * 100,
#    "The capital of France is",
#    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
#llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8", trust_remote_code=True)
llm = LLM(model="/home/gnap/Models/Meta-Llama-3-8B-Instruct-FP8", trust_remote_code=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(time.time() - start)
