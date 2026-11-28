import argparse
import os
import time
from configs import *
from generator import Generator

DIRS = {"home": "runs/", "hpc": "/scratch/an4245/runs"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="home", choices=DIRS)
    parser.add_argument("--answer-des", nargs="+", default=["task"])
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--model-name", default="OLMo-2-1124-7B-Instruct")
    parser.add_argument("--model-origin", default="allen")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    root = DIRS[args.dir]
    m = ModelConfig(origin=args.model_origin, name=args.model_name)

    gen = Generator(m)
    if args.judge:
        res = load(root)
        msgs = args.answer_des[0].split(",")
        dcs = [JudgeConfig(msgs, **{k: c["Data Config"][k] for k in ("name", "split", "max_idx", "answer_des")})
               for c in res["config"]]
        gcs = [GenConfig(regex_name=dc.name) for dc in dcs]
        inputs, outputs = gen.generate([dc.prompts(io["Outputs"]) for dc, io in zip(dcs, res["io"], strict=True)], gcs)
        acc = [dc.score(o) for dc, o in zip(dcs, outputs, strict=True)]
        save(os.path.join(root, "judged", time.strftime("%Y-%m-%d_%H-%M-%S")),
             m, dcs, gcs, inputs, outputs, acc, names=res["file"])
    else:
        dcs = [PredConfig(ad.split(","), max_idx=args.max_val) for ad in args.answer_des]
        gcs = [GenConfig() for _ in dcs]
        inputs, outputs = gen.generate([dc.prompts() for dc in dcs], gcs)
        save(root, m, dcs, gcs, inputs, outputs)