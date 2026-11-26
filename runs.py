FMT = """<format>
Format your response exactly as follows:
Reason: <your reasoning>
Answer: <your answer>
</format>
"""


TASK = """<task>
Think through the following question, step by step. 
Your answer should be {answer_des}.
</task>
<question>
{question}
</question>
"""

REWRITE = """Rewrite the following response in the required format. Do not change the answer.
<response>
{response}
</response>
"""

JUDGE_MSG = """Given a task and a response, extract the answer from the response.
The answer should be in the form described by the task, but should not be modified
in any other way. Do not solve the original task. Do not use outside knowledge. 
<task>
Think through the following question, step by step. 
Your answer should be {answer_des}.
</task>
<response>
{response}
</response>
"""

from dataclasses import dataclass
from datasets import Dataset, load_dataset
import pickle
import os

@dataclass
class RunInfo():
    model_name: str
    inputs: list[str] = None
    outputs: list[str] = None
    max_tokens: int = 100
    max_model_len: int = 8192
    temp: float = 0.0
    so: bool = False
    regex: str = None
    answer_des: str = None
    acc: float = None

    


class Run():
    @classmethod
    def load_from_path(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __init__(self, gen_model_name: str, judge_model_name: str, save_path: str = None):
        self.save_path = save_path
        self.gen = RunInfo(gen_model_name)
        self.judge = RunInfo(judge_model_name)

    def save(self):
        if self.save_path is not None:
            with open(self.save_path, "wb") as f:
                pickle.dump(self, f)


    def get_info_from_ds(self, ds: Dataset):
        return ds["answer"], ds["question"]

    def add_gen_inputs(self, ds: Dataset):
        self.judge.golds, qs  = self.get_info_from_ds(ds)
        self.save()
        self.gen.inputs = [[{"role": "user", "content": TASK.format(answer_des = self.gen.answer_des, question=q)}] for q in qs]
        self.save()

    def add_gen_outputs(self, outputs: list[str]):
        self.gen.outputs = [o.outputs[0].text for o in outputs]
        self.save()
        self.judge.inputs = [[{"role": "user", "content": JUDGE_MSG.format(answer_des = self.judge.answer_des, response=r)}] for r in self.gen.outputs]
        self.save()

    def add_judge_outputs(self, outputs: list[str]):
        self.judge.outputs = [o.outputs[0].text for o in outputs]
        self.save()
        self.acc = sum([self.compare(o, g) for o, g in zip(self.judge.outputs, self.judge.golds)])
        self.save()


    @staticmethod
    def compare(pred: str, gold: str) -> bool:
        pass



class GSMRun(Run):
    def __init__(self, gen_model_name: str, judge_model_name: str, save_path: str):
        super().__init__(gen_model_name, judge_model_name, save_path)
        self.gen.answer_des = "a single integer"
        self.judge.answer_des =  "a single integer"
        self.judge.max_tokens = 20
    
    @staticmethod
    def compare(pred, gold):
        return float(pred) == float(gold.strip().replace(",", "").split("#### ")[-1])


def judge_or_generate(run: Run, mode: str, gpu_mem_util: float = 0.3):
    info = run.judge if mode=="judge" else run.gen

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(info.model_name, max_model_len = info.max_model_len, gpu_memory_utilization=gpu_mem_util)
    params = SamplingParams(temperature=info.temp, 
                            max_tokens = info.max_tokens,
                            structured_outputs = StructuredOutputsParams(regex=info.regex) if info.so else None)
    outputs = llm.chat(info.inputs, params)
    run.add_judge_outputs(outputs) if mode=="judge" else run.add_gen_outputs(outputs)



if __name__=="__main__":
    gr1 = GSMRun.load_from_path("run.pkl")
    judge_or_generate(gr1, "judge")

    with open("run.pkl", "rb") as f: 
        obj = pickle.load(f)
        print(obj.judge.outputs) 



    


