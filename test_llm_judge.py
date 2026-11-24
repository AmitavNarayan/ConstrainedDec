from transformers import AutoModelForCausalLM
from datasets import load_dataset
from vllm import LLM, SamplingParams

model_path = "/scratch/an4245/OLMo-2-1124-7B-Instruct"
ds_name, splits = "QAEval", ["test", "train", "val"]
max_tokens = 100
prompt = """Extract the final answer from the model response below.

Return only the extracted answer. Do not solve the original question, evaluate whether the answer is correct, or use outside knowledge.
<response>
{response}
</response>
Extracted answer: """

llm = LLM(model_path)
ds = load_dataset("json", data_files=[f"/scratch/an4245/{ds_name}/{x}.json" for x in splits], split="main")
outputs = llm.chat([[{"role": "user", "content": prompt.format(ex["response"])}] for ex in ds], sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
score = sum(int(ex["extracted_answer"]==y.outputs[0].text.strip()) for ex, y in zip(ds, outputs)) / len(outputs)
print(score)

    

