
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from utils.formatting import ds_formats,  INSTRUCTION

class ProcessDS():
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    def get_str(self, example: dict, ds_name: str) -> dict:
        user_content=INSTRUCTION.format(**ds_formats[ds_name], question=example["question"])
        prompts = self.tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
        gold = int(example["answer"].split("####")[-1].strip().replace(",", ""))
        return {"prompts": prompts, "gold": gold}

    def make_ds(self, dataset: DatasetDict,
                ds_name: str,
                save_path: str,
                num_cpus: int = 24):
        dataset.map(self.get_str, fn_kwargs={"ds_name": ds_name}, num_proc=num_cpus, remove_columns=dataset["train"].column_names).save_to_disk(f"{save_path}")


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct")
    gsm = load_dataset("openai/gsm8k", "main")
    ProcessDS(tokenizer).make_ds(gsm, "gsm", "/scratch/an4245/gsm8k_format")