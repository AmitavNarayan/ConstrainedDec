

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from datasets import Dataset
from torch.utils.data import DataLoader
import wandb
from datasets import load_from_disk
from utils.scorers import Scorer



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
                 max_new_tokens: int = 200,
                 load_dir: str = None):
        self.m = model
        self.device = self.m.device
        self.m.requires_grad_(False)
        self.head = Head(self.m.config.hidden_size, self.device, head_dtype)
        if load_dir is not None:  self.head.load_state_dict(torch.load(load_dir))
        self.m.model.layers[-1].register_forward_hook(self.head.hook)
        self.head_optim = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.max_new_tokens = max_new_tokens
        self.scorer = Scorer()
        self.tokenizer = tokenizer
        
    def collate_fn_train(self, batch):
        new_batch = {"gold": [], "input_ids": [], "mask": [], "labels": []}
        maxlen = max([len(ex["prompt_ids"] + ex["target_ids"]) for ex in batch])
        for ex in batch:
            input_ids = ex["prompt_ids"] + ex["target_ids"]
            pad_len = maxlen - len(input_ids)
            new_batch["input_ids"].append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            new_batch["mask"].append([True] * len(input_ids) + [False] * pad_len)
            new_batch["labels"].append([-100] * len(ex["prompt_ids"]) + ex["target_ids"] + [-100] * pad_len)
            new_batch["gold"].append(ex["gold"])
        for key in ["mask", "input_ids", "labels"]: new_batch[key] = torch.tensor(new_batch[key])
        return new_batch

    def collate_fn_eval(self, batch):
        new_batch = {"gold": [], "input_ids": [], "mask": []}
        maxlen = max([len(ex["prompt_ids"]) for ex in batch])
        for ex in batch:
            ids = ex["prompt_ids"]
            pad_len = maxlen - len(ids)
            new_batch["input_ids"].append([self.tokenizer.pad_token_id] * pad_len + ids)
            new_batch["mask"].append([False] * pad_len + [True] * len(ids))
            new_batch["gold"].append(ex["gold"])
        for key in ["mask", "input_ids"]: new_batch[key] = torch.tensor(new_batch[key])
        return new_batch

    def train(self, ds: Dataset, save_dir: str, batch_size: int=32):
        run = wandb.init(project="Constrained_dec")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn_train)
        for batch in dl:
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            output = self.m(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = output.loss
            self.head_optim.zero_grad()
            loss.backward()
            self.head_optim.step()
            run.log({"train-loss": loss.item()})
        torch.save(self.head.state_dict(), save_dir)
        run.finish()
        return

    def eval(self, ds: Dataset, batch_size: int=32):
            run = wandb.init(project="Constrained_dec")
            dl = DataLoader(ds, batch_size=batch_size, collate_fn=self.collate_fn_eval)
            score = 0
            in_format = 0
            for batch in dl:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["mask"].to(self.device)
                prefix_len = input_ids.shape[1]
                gold = batch["gold"]
                output = self.m.generate(input_ids=input_ids, attention_mask=mask, do_sample=False, max_new_tokens=self.max_new_tokens)
                for i, ex in enumerate(output):
                    text = self.tokenizer.decode(ex[prefix_len:], skip_special_tokens=True)
                    score += self.scorer.score(text, gold[i], "int")
                    in_format += self.scorer.check_in_format(text)

            score = score / len(ds)
            in_format = in_format / len(ds)
            run.log({"score": score, "frac_in_format": in_format})
            run.finish()
            return

        
       
if __name__=="__main__":
    model = AutoModelForCausalLM.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct", device_map="auto", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("/scratch/an4245/Qwen2.5-Instruct-Tokenizer")
    num_train = 5000
    num_test = 250
    
    ds = load_from_disk("/scratch/an4245/gsm8k_format")

    math = ConstrainedDec(model, tokenizer)
    math.eval(ds=ds["test"].select(range(num_test)))
    math.train(ds=ds["train"].select(range(num_train)), save_dir="/scratch/an4245/Qwen2.5-7-head")
    math.eval(ds=ds["test"].select(range(num_test)))


