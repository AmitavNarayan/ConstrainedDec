
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset, load_from_disk
import wandb
from utils.formatting import ds_formats, FMT, TASK, REWRITE, acc
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import re

       


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
        output = self.forward(output).to(output.dtype)
        return output

def score(ans: str, gold: int, metric: str = "gsm"):
        if metric == "acc": 
            result = re.findall(acc, ans)
            if result: result = result[-1]
        else: 
            result = re.fullmatch(ds_formats[metric]["answer_format"], ans.strip())
            if result: result = result[2]
        val = 0
        if result: val = int(float(gold) == int(result.replace(",", "")))
        return val

    
class ConstrainedDec():
    def __init__(self, tokenizer_path: str = "/scratch/an4245/Qwen2.5-Instruct-Tokenizer",
                  model_path: str = "/scratch/an4245/Qwen2.5-7B-Instruct",
                 head_dtype: torch.dtype = torch.float32,
                 lr: float = 5e-4,
                 max_new_tokens: int = 512,
                 load_dir: str = None,
                 num_gpus: int=2):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.llm = LLM(model_path, tensor_parallel_size=num_gpus)


    
    def eval(self, ds: Dataset, ds_name: str = "gsm", batch_size: int=3):
            run = wandb.init(project="Constrained_dec", group="two_turn_test")
            ds_fmt = ds_formats[ds_name]
            llm_input = lambda prompts, so = StructuredOutputsParams(regex=ds_fmt["answer_format"]): {
                  "messages": [[{"role": "user", "content": p}] for p in prompts],
                  "sampling_params": SamplingParams(max_tokens=self.max_new_tokens, 
                                    temperature=0, 
                                    structured_outputs=so)
            }
            qs = ds["question"]
            turn_one_prompts = [TASK.format(question=q, **ds_fmt) for q in qs]
            turn_one_output = self.llm.chat(**llm_input(turn_one_prompts, None))
            turn_two_prompts = [(FMT+TASK+REWRITE).format(question=q, response=r.outputs[0].text, **ds_fmt) for r,q in zip(turn_one_output, qs)]
            turn_two_output = self.llm.chat(**llm_input(turn_two_prompts))
            std_prompts = [(FMT+TASK).format(question=q, **ds_fmt) for q in qs]
            std_output = self.llm.chat(**llm_input(std_prompts))
            std_no_cd_prompts = [(FMT+TASK).format(question=q, **ds_fmt) for q in qs]
            std_no_cd_output = self.llm.chat(**llm_input(std_no_cd_prompts, None))
            std_no_cd_correct = 0
            two_turn_correct = 0
            std_correct = 0
            
            for a, x, y, z in zip(ds["answer"], turn_two_output, std_output, std_no_cd_output):
                gold = a.split("####")[-1].strip().replace(",", "")
                two_turn_correct += score(x.outputs[0].text, gold)
                std_correct += score(y.outputs[0].text, gold)
                std_no_cd_correct += score(z.outputs[0].text, gold, "acc")
            run.log({"frac_correct_standard_no_cd": std_no_cd_correct / ds.num_rows})
            run.log({"frac_correct_standard": std_correct / ds.num_rows})
            run.log({"frac_correct_two_turn": two_turn_correct / ds.num_rows})
            


        
if __name__=="__main__":
    ds = load_from_disk("/scratch/an4245/gsm8k")["test"]
    ConstrainedDec().eval(ds=ds, batch_size=128)
   

    


