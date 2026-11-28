from dataclasses import dataclass, asdict
from statistics import mean
import json
import os
import time
from datasets import load_dataset

INDENT = 4
ORIGINS = {"mlx": "mlx-community/", "allen": "allenai/", "qwen": "Qwen/", "scratch": "/scratch/an4245/"}
REGEX = {"gsm": r"-?\d+(\.\d+)?"}


def gsm_score(preds: list[str], ans: list[str]) -> float:
    return mean([float(p) == float(a.strip().replace(",", "").split("#### ")[-1])
                 for (p, a) in zip(preds, ans, strict=True)])


SCORERS = {"gsm": gsm_score}

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


@dataclass(frozen=True)
class ModelConfig:
    name: str = None
    origin: str = None
    max_model_len: int = None
    gpu_mem_util: float = 0.9

    @property
    def path(self) -> str:
        return ORIGINS[self.origin] + self.name


@dataclass(frozen=True)
class GenConfig:
    max_tokens: int = 512
    temp: float = 0.0
    regex_name: str = None

    @property
    def regex(self) -> str | None:
        return REGEX[self.regex_name] if self.regex_name else None


@dataclass(frozen=True)
class DataConfig:
    msgs: list[str]
    q_name: str = "question"
    a_name: str = "answer"
    name: str = "gsm"
    ds_name: str = "apple/GSM-Symbolic"
    split: str = "test"
    max_idx: int = None
    answer_des: str = "a single integer"

    def __post_init__(self):
        ds = load_dataset(self.ds_name, split=self.split)
        if self.max_idx:
            ds = ds.select(range(self.max_idx))
        object.__setattr__(self, "inst", "".join(INST[msg] for msg in self.msgs))
        object.__setattr__(self, "qs", list(ds[self.q_name]))
        object.__setattr__(self, "ans", list(ds[self.a_name]))

    def score(self, preds: list[str]) -> float:
        return SCORERS[self.name](preds, self.ans)


def save(root, m, dcs, gcs, inputs, outputs, acc: list[float] | None = None, names: list[str] | None = None):
    os.makedirs(root, exist_ok=True)
    curr_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    for k, (dc, gc, i, o) in enumerate(zip(dcs, gcs, inputs, outputs, strict=True)):
        name = names[k] if names else f"{curr_time}-config-{k}.json"
        with open(os.path.join(root, name), "w") as f:
            json.dump({"Model Config": asdict(m), "Data Config": asdict(dc),
                       "Generation Config": asdict(gc)}, f, indent=INDENT)
        with open(os.path.join(root, name.replace("-config-", "-io-")), "w") as f:
            json.dump({"Inputs": i, "Outputs": o,
                       "Accuracy": acc[k] if acc is not None else None}, f, indent=INDENT)


def load(root):
    configs = [f for f in os.listdir(root) if "-config-" in f]
    loader = lambda f: json.load(open(os.path.join(root, f)))
    res = {"file": configs,
           "config": [loader(c) for c in configs],
           "io": [loader(c.replace("-config-", "-io-")) for c in configs]}
    return res