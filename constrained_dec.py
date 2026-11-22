

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from datasets import Dataset
from torch.utils.data import DataLoader
import wandb
from datasets import load_from_disk
from utils.score import score
import outlines
from outlines.types import Regex
from utils.formatting import ds_formats



class Head(torch.nn.Module):
    def __init__(self, d: int, device: str = "auto", dtype: torch.dtype = torch.float32):
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


    
class ConstrainedDec():
    def __init__(self, model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 head_dtype: torch.dtype = torch.float32,
                 lr: float = 5e-4,
                 max_new_tokens: int = 512,
                 load_dir: str = None):
        self.model = model
        self.device = self.model.device
        self.model.requires_grad_(False)
        self.head = Head(self.model.config.hidden_size, self.device, head_dtype)
        if load_dir is not None:  self.head.load_state_dict(torch.load(load_dir))
        self.model.model.layers[-1].register_forward_hook(self.head.hook)
        self.head_optim = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        
    
    def eval(self, ds: Dataset, ds_name: str = "gsm", batch_size: int=32):
            run = wandb.init(project="Constrained_dec")
            dl = DataLoader(ds, batch_size=batch_size)
            correct = 0
            c_model = outlines.from_transformers(self.model, self.tokenizer)
            for batch in dl:
                prompts = batch["prompts"]
                gold = batch["gold"]
                output = c_model.batch(prompts, Regex(ds_formats[ds_name]["answer_format"]), max_new_tokens=self.max_new_tokens, do_sample=False)
                for i, text in enumerate(output):
                    correct += score(text, gold[i])
            run.log({"frac_correct": correct / len(ds)})
            run.finish()
            return

        
       
if __name__=="__main__":
    model = AutoModelForCausalLM.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct", device_map="cuda", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("/scratch/an4245/Qwen2.5-Instruct-Tokenizer")
    num_train = 5000
    num_test = 1000
    
    ds = load_from_disk("/scratch/an4245/gsm8k_format")

    math = ConstrainedDec(model, tokenizer)
    math.eval(ds=ds["test"], batch_size=128)


