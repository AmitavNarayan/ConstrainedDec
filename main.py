from configs import *
from generator import Generator
import json
import wandb
import sys
if __name__=="__main__":
    m = ModelConfig(model_path=sys.argv[1])
    dcs, gcs = [DataConfig(["task"])], [GenConfig()]
    qs  = [dc.qs for dc in dcs]
    preds = Generator(m).predict(dcs, gcs, qs)
    save(m, dcs, gcs, qs, preds)

    
