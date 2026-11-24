import torch
from datasets import Dataset, load_dataset
import wandb
from utils.formatting import FMT, TASK, REWRITE, JUDGE_MSG, ds_fmts
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import argparse

       


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


class ConstrainedDec():
    def __init__(self, model_path: str, 
                 ds: Dataset,
                 ds_name: str,
                 ms: str,
                 run_name: str,
                 max_new_tokens: int = 1024,
                 max_model_len: int = 4096):
        self.llm = LLM(model_path, max_model_len=max_model_len)
        self.run = wandb.init(project="Constrained_dec", name=f"{run_name}")
        self.qs, self.golds = ds["question"], ds["answer"]
        self.max_new_tokens = max_new_tokens
        self.ds_fmt = ds_fmts[ds_name]
        self.ds_name = ds_name
        self.ms = ms


    def score(self, responses: list[str], tol: float = .01, eps: float = 10e-7):
        msgs = self.make_msgs(JUDGE_MSG, {"response": responses})
        extracted_answers = self.generate(msgs, so=StructuredOutputsParams(regex=self.ds_fmt["regex_fmt"]))
        score = 0
        for ea, g in zip(extracted_answers, self.golds):
            ea = ea.strip()
            if self.ds_name == "strategy":
                 score += ea.lower() == g.lower()
            elif self.ds_name == "aime":
                 score += int(ea) == int(g)
            elif self.ds_name == "medqa": 
                 score += ea == g
            elif self.ds_name == "finqa": 
                if "%" in ea: 
                    ea = ea.replace("%", "")
                    ea = float(ea) / 100
                else: ea = float(ea)
                score += abs((float(ea) - float(g)) / (eps+float(g))) < tol
            
        score = score / len(self.golds)
        return score
         
         
    def make_msgs(self, inst: str, format_args: dict[str, list]):
            fmts = [dict(zip(format_args, vals)) | self.ds_fmt for vals in zip(*format_args.values())]
            msgs=[[{"role": "user", "content": inst.format(**x)}] for x in fmts]
            return msgs

        
    def generate(self, msgs: list[list[dict[str, str]]], 
                 so: StructuredOutputsParams = None):  
        outputs = self.llm.chat(messages=msgs, 
            sampling_params=SamplingParams(max_tokens=self.max_new_tokens, temperature=0, structured_outputs=so))
        ret = [o.outputs[0].text for o in outputs]
        return ret
    
    def eval(self):
            output = {
                "fmt+task": self.generate(self.make_msgs(FMT+TASK, {"question": self.qs})),
                "task": self.generate(self.make_msgs(TASK, {"question": self.qs}))
            }
            scores = {
                f"{self.ds_name}-{self.ms}-{key}": self.score(output[key])
                 for key in output
            }
            self.run.log(scores)
            wandb.finish()

            
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model-stage")
    parser.add_argument("--model-name")
    
    args = parser.parse_args()
    ds_name, ms, mn = args.dataset, args.model_stage, args.model_name
    max_q_chars = 8000

    if ds_name=="strategy": 
         ds = load_dataset("voidful/strategyqa", split="train")
         ds = ds.map(lambda ex: {"answer": str(ex["answer"])})
    if ds_name=="aime": 
         ds = load_dataset("gneubig/aime-1983-2024", split="train")
         ds = ds.rename_column("Question", "question")
         ds = ds.rename_column("Answer", "answer")
    if ds_name=="medqa": 
         ds = load_dataset("medalpaca/medical_meadow_medqa", split="train")
         ds = ds.rename_column("input", "question")
         ds = ds.map(lambda ex: {"answer": ex["output"][0]},remove_columns="output")
    if ds_name=="finqa": 
         ds = load_dataset("ChanceFocus/flare-finqa", split="train")
         ds = ds.filter(lambda ex: ex["answer"] not in ["yes", "no"])
         ds = ds.rename_column("query", "question")
    ds = ds.select_columns(["question", "answer"])
    ds = ds.filter(lambda ex: len(ex["question"]) <= max_q_chars)
    
        

    if mn=="tulu": model_path = f"/scratch/an4245/Llama-3.1-Tulu-3-8B-{ms}" 
    if mn=="olmo": model_path = f"/scratch/an4245/OLMo-2-1124-7B-{ms}"

    cd = ConstrainedDec(model_path=model_path, ds=ds, ms=ms, ds_name=ds_name, run_name=f"{mn}")
    cd.eval()
    
   