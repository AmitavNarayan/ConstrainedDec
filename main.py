from configs import *
from generator import Generator
import json
import wandb
import sys
if __name__=="__main__":
    m = ModelConfig()
    dcs, gcs = [DataConfig(["task"], indices=range(50))], [GenConfig()]
    qs  = [dc.qs for dc in dcs]
    preds = Generator(m).predict(dcs, gcs, qs)
    save("personal", m, dcs, gcs, qs, preds)

    
