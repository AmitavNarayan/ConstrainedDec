import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset, load_from_disk
import wandb
from utils.formatting import ds_formats, FMT, TASK, REWRITE, acc
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import argparse
import re
from math_verify 

       


class Head(torch.nn.Module):
    def __init__(self, d: int, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.f = torch.nn.Linear(d,d, device=device, dtype=dtype)
        torch.nn.init.zeros_(self.f.weight)
        torch.nn.init.zeros_(self.f.bias)
    def forward(self, x: torch.Tensor):
        x = x.to(self.f.weight.dtype)
        y = x + self.f(x)
        return y
    def hook(self, module: torch.nn.Module, inputs: torch.Tensor, output: torch.Tensor):
        if isinstance(output, tuple): 
            ret = (self.forward(output[0]).to(output[0].dtype), output[1])
        else: 
            ret = self.forward(output).to(output.dtype)
        return ret

def add_head_to_vllm(m: LLM, head_path: str):
     head = Head(m.config.hidden_size)
     head.load_state_dict(torch.load(head_path, map_location="cuda"))
     handle = m.model.norm.register_forward_hook(head.hook)
     return handle
     


def score(ans: str, gold: str, metric: str):
        val = 0
        ans = ans.strip()
        if metric == "gpqa":
            final_ans = ans[-1]
            if final_ans == "gold": val = 1
        if metric == "aime"
            final_ans = re.findall(r'-?\d+', ans)[-1]
        else: 
            result = re.fullmatch(ds_formats[metric]["answer_format"], ans.strip())
            if result: result = result[2]
        val = 0
        if result: val = int(float(gold) == float(result.replace(",", "")))
        return val

class ConstrainedDec():
    def __init__(self, tokenizer_path: str = "/scratch/an4245/Qwen2.5-Instruct-Tokenizer",
                  model_path: str = "/scratch/an4245/Qwen2.5-7B-Instruct",
                 lr: float = 5e-4,
                 max_new_tokens: int = 512,
                 model_device: str = "cuda:0",
                 model_dtype: torch.dtype = torch.bfloat16,
                 head_path: str = None,
                 vllm_mem_util: float = 0.9):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.llm = LLM(model_path, gpu_memory_utilization = vllm_mem_util, enforce_eager=True, compilation_config={"mode": 0})
        self.md = model_device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=self.md, dtype=model_dtype)
        self.model.requires_grad_(False)
        self.head = Head(self.model.config.hidden_size, device=self.md)
        self.model.model.norm.register_forward_hook(self.head.hook)
        self.head_path = head_path
        if self.head_path is not None: self.head.load_state_dict(torch.load(head_path))
        self.head_optim = torch.optim.AdamW(self.head.parameters(), lr = lr)

    def make_prompts(self, inst: str, format_args: dict[str, list] = {}):
            format_args.update(self.qs)
            fmts = [dict(zip(format_args, vals)) | self.ds_fmt for vals in zip(*format_args.values())]
            prompts=[[{"role": "user", "content": inst.format(**x)}] for x in fmts]
            return prompts

    def generate(self, prompts: list[list[dict[str, str]]], 
                 so: StructuredOutputsParams = None, 
                 ret_tokens: bool = False, 
                 use_head: bool = False):   
        hp = self.head_path
        if use_head: self.llm.apply_model(lambda m: setattr(m, "hook", add_head_to_vllm(m, hp)))
        output = self.llm.chat(messages=prompts, 
            sampling_params=SamplingParams(max_tokens=self.max_new_tokens, temperature=0, structured_outputs=so))
        if use_head: self.llm.apply_model(lambda m: m.hook.remove())
        ret = [x.outputs[0].token_ids for x in output] if ret_tokens else [x.outputs[0].text for x in output]
        return ret

    def collate_fn_train(self, batch):
        new_batch = {"input_ids": [], "target_mask": [], "prompt_mask": []}
        maxlen = max([len(ex["prompt_ids"] + ex["target_ids"]) for ex in batch])
        for ex in batch:
            input_ids = ex["prompt_ids"] + ex["target_ids"]
            pad_len = maxlen - len(input_ids)
            new_batch["input_ids"].append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            new_batch["prompt_mask"].append([True] * len(input_ids) + [False] * pad_len)
            new_batch["target_mask"].append([False] * (len(ex["prompt_ids"]) - 1) + [True] * len(ex["target_ids"]) + [False] * pad_len)
        for key in ["prompt_mask", "target_mask", "input_ids"]: new_batch[key] = torch.tensor(new_batch[key])
        return new_batch


    def train(self, ds: Dataset, ds_name: str, batch_size: int=32, head_path: str = None):
        
        return

    def eval(self, ds: Dataset, ds_name: str = "gsm", run_name: str = None):
            self.ds_fmt, self.qs = ds_formats[ds_name], {"question": ds["question"]}
            run = wandb.init(project="Constrained_dec", name=run_name)
            so = StructuredOutputsParams(regex=self.ds_fmt["answer_format"])
            output = {}
            output["fmt+task"] = self.generate(self.make_prompts(FMT+TASK))
            output["task"] = self.generate(self.make_prompts(TASK))
            ans = [a.split("####")[-1].strip().replace(",", "") for a in ds["answer"]]
            vals = {
                k: sum(score(o, g, "acc") for o, g in zip(v, ans)) / len(ans)
                for k, v in output.items()
            }
            run.log(vals)
            wandb.finish()

            
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model-type")
    parser.add_argument("--model-name")
    args = parser.parse_args()
    d, mt, mn = args.dataset, args.model_type, args.model_name
    stem = "/scratch/an4245/"
    ds = load_from_disk(f"{stem}{d}")
    if mn=="tulu": m = f"{stem}Llama-3.1-Tulu-3-8B-{mt}" 
    if mn=="olmo": m = f"{stem}OLMo-2-1124-7B-{mt}"
    cd = ConstrainedDec(m, m)
    cd.eval(ds=ds["train"], ds_name=d, run_name=f"{mt}-{mn}")
    
   