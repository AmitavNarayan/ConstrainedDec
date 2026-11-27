def gsm_score(preds: list[str], ans: list[str]) -> float:
    return mean([float(p) == float(a.strip().replace(",", "").split("#### ")[-1]) for (p, a) in zip(preds, ans)])
SCORERS = {"gsm": gsm_score}

REGEX = {"gsm": r"-?\d+(\.\d+)?"}
ROOT = "/scratch/an4245"
INST = {
    "judge": """Given a task and a response, extract the answer from the response.
    The answer should be in the form described by the task, but should not be modified
    in any other way. Do not solve the original task. Do not use outside knowledge. 
    <task>
    Think through the following question, step by step. 
    Your answer should be {answer_des}.
    </task>
    <response>
    {response}
    </response>
    """,

    "format": """<format>
    Format your response exactly as follows:
    Reason: <your reasoning>
    Answer: <your answer>
    </format>
    """,

    "task": """<task>
    Think through the following question, step by step. 
    Your answer should be {answer_des}.
    </task>
    <question>
    {question}
    </question>
    """}


from dataclasses import dataclass, asdict
from datasets import load_dataset
from statistics import mean




@dataclass(frozen=True)
class ModelConfig():
    name = "OLMo-2-1124-7B-Instruct"
    model_name: str = None
    max_model_len: int = 8192
    gpu_mem_util: float = 0.3
    def __post_init__(self):
        object.__setattr__(self, "model_name", f"{BASE_DIR}/{name}")

@dataclass(frozen=True)
class GenConfig():
    max_tokens: int = 512
    temp: float = 0.0
    regex: str = None
    def __post_init__(self):
        if self.regex: object.__setattr__(self, "regex", REGEX[self.regex])

@dataclass(frozen=True)
class DataConfig():
    msgs: list[str]
    q_name: str = "question"
    a_name: str = "answer"
    name: str = "gsm"
    ds_name: str = "apple/GSM-Symbolic"
    split: str = "test"
    indices: list = range(5000)
    answer_des: str = "a single integer"

    def __post_init__(self):
        ds = load_dataset(self.ds_name, split=self.split).select(self.indices)
        object.__setattr__(self, "inst", "".join([INST[msg] for msg in self.msgs]))
        object.__setattr__(self, "qs", ds[self.q_name])
        object.__setattr__(self, "ans", ds[self.a_name])

    def score(self, preds: list[str]) -> float:
        return SCORERS[self.name](preds, self.ans)

def save(m: ModelConfig, dcs: list[DataConfig], gcs: list[GenConfig], qs: list[list[str]], preds: list[list[str]]):
    save_dir = f"{ROOT}/{m.model_name}"
    for dc, gc, q_list, pred_list in zip(dcs, gcs, qs, preds):
        with open(f"{save_dir}/{dc.name}/{dc.answer_des}", "w") as f:
            json.dump([asdict(m), asdict(dc), asdict(gc), q_list, pred_list], f)

