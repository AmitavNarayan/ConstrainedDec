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

    if args.judge:
        res = load(root)
        msgs = args.answer_des[0].split(",")
        dcs = [DataConfig(msgs, **{k: v for k, v in c["Data Config"].items() if k != "msgs"})
               for c in res["config"]]
        gcs = [GenConfig(regex_name=dc.name) for dc in dcs]
        inputs, outputs, acc = Generator(m).judge(dcs, gcs, [io["Outputs"] for io in res["io"]])
        save(os.path.join(root, "judged", time.strftime("%Y-%m-%d_%H-%M-%S")),
             m, dcs, gcs, inputs, outputs, acc, names=res["file"])
    else:
        dcs = [DataConfig(ad.split(","), max_idx=args.max_val) for ad in args.answer_des]
        gcs = [GenConfig() for _ in dcs]
        inputs, outputs = Generator(m).predict(dcs, gcs, [dc.qs for dc in dcs])
        save(root, m, dcs, gcs, inputs, outputs)