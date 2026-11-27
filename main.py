from configs import *
from generator import Generator
import json
import wandb
import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="hpc")
    parser.add_argument("--answer-descriptions", default="task")
    parser.add_argument("model-name", default="OLMo-2-1124-7B-Instruct")
    parser.add_argument("model-origin", default="allen")
    args = parser.parse_args()
    m = ModelConfig(origin=args.model_origin, name=args.model_name)
    dcs, gcs = [DataConfig(args.answer_descriptions)], [GenConfig()]
    qs  = [dc.qs for dc in dcs]
    preds = Generator(m).predict(dcs, gcs, qs)
    save(args.save_dir, m, dcs, gcs, qs, preds)

    
