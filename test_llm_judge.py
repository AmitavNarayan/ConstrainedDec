
import json
from vllm import LLM, SamplingParams

model_path = "/scratch/an4245/OLMo-2-1124-7B-Instruct"
ds_name, splits = "QAEval", ["test", "train", "val"]
max_tokens = 100
prompt = """Extract the final answer from the model response below.

Return the extracted answer without any other text. Do not solve the original question.
Do not modify the formatting, wording, order, or explanation of the answer.
<question>
{question}
<response>
{response}
</response>
Extracted answer: """

llm = LLM(model_path)
ds = [ex for s in splits for ex in json.load(open(f"/scratch/an4245/{ds_name}/{s}.json")).values()]
outputs = llm.chat([[{"role": "user", "content": prompt.format(response=ex["response"], question=ex["query"])}] for ex in ds], sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
score = 0
for ex, o in zip(ds, outputs):
    x, y = ex["extracted_answer"], o.outputs[0].text.strip()
    print(f"Gold:{x}")
    print(f"Pred:{y}")
    score += int(x==y)
    print()
score = score / len(outputs)
print(score)


    

