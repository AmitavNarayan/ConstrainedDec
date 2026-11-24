
import json
from vllm import LLM, SamplingParams

model_path = "/scratch/an4245/OLMo-2-1124-7B-Instruct"
ds_name, splits = "QAEval", ["test", "train", "val"]
max_tokens = 100
prompt = """Extract the final answer from the model response below.

Return the extracted answer without any other text. Return the extracted answer without any other text. 
Return the extracted answer without any other text. Return the extracted answer without any other text. 
<response>
{response}
</response>
Extracted answer: """

llm = LLM(model_path)
ds = [ex for s in splits for ex in json.load(open(f"/scratch/an4245/{ds_name}/{s}.json")).values()]
outputs = llm.chat([[{"role": "user", "content": prompt.format(response=ex["response"])}] for ex in ds], sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
score = sum(int(ex["extracted_answer"]==y.outputs[0].text.strip()) for ex, y in zip(ds, outputs)) / len(outputs)
print(score)

    

