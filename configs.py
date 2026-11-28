from dataclasses import dataclass, asdict
from statistics import mean
import json
import os
import time
from datasets import load_dataset

INDENT = 4
ORIGINS = {"mlx": "mlx-community/", "allen": "allenai/", "qwen": "Qwen/", "scratch": "/scratch/an4245/"}


def gsm_score(preds: list[str], ans: list[str]) -> float:
    return mean([float(p) == float(a.strip().replace(",", "").split("#### ")[-1])
                 for (p, a) in zip(preds, ans, strict=True)])


INFO = {
    "gsm": {
        "ds_name": "apple/GSM-Symbolic",
        "q_col_name": "question",
        "a_col_name": "answer",
        "answer_des": "a single integer",
        "score": gsm_score,
        "regex": r"-?\d+(\.\d+)?",
        "examples": [
            "Reason: 20% of $1,250 is $250.2342. Answer: $^298.#32. -> 298",
            "Fiona gives away 3 apples after receiving 9, leaving her with -> 9",
            "Finally, since 3*4=twelve, we have our final answer. -> 12",
        ],
    }
}


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
    "advanced_judge": """ <task> 
    Your task is to extract the final answer from a model's response to a problem. 
    You must output the final answer of the response, in precisely the format described below.
    You must ignore any intermediate results, any partial solutions, and any information from the problem statement. 
    Do not solve the problem or reason about the answer in any way. 
    Return the response's final answer even if it is wrong. 
    If there is no final answer, return the last possible valid answer.
    Your extraction should be in the format {answer_des}.
    </task>
    <examples>
    {examples}
    </examples>
    The model response is listed below. 
    <response>
    {response}
    </response>
    Your extracted answer:
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
        return INFO[self.regex_name]["regex"] if self.regex_name else None


@dataclass(frozen=True)
class PredConfig:
    msgs: list[str]
    name: str = "gsm"
    split: str = "test"
    max_idx: int = None
    answer_des: str = None

    def __post_init__(self):
        info = INFO[self.name]
        if self.answer_des is None:
            object.__setattr__(self, "answer_des", info["answer_des"])
        ds = load_dataset(info["ds_name"], split=self.split)
        if self.max_idx:
            ds = ds.select(range(self.max_idx))
        object.__setattr__(self, "inst", "".join(INST[msg] for msg in self.msgs))
        object.__setattr__(self, "qs", list(ds[info["q_col_name"]]))


@dataclass(frozen=True)
class JudgeConfig:
    msgs: list[str]
    name: str = "gsm"
    split: str = "test"
    max_idx: int = None
    answer_des: str = None

    def __post_init__(self):
        info = INFO[self.name]
        if self.answer_des is None:
            object.__setattr__(self, "answer_des", info["answer_des"])