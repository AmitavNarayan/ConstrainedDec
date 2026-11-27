from configs import *
from generator import Generator
import json
import wandb
import sys
if __name__=="__main__":
    model_name = sys.argv[1]
    m = ModelConfig(model_name=model_name)
    dcs, gcs, qs  = [DataConfig(["task"])], [GenConfig()], [dc.qs for dc in dcs]
    preds = Generator(m).predict(dcs, gcs, qs)
    save(m, dcs, gcs, qs, preds)
    dcs = [DataConfig(["task", "fmt"])]
    preds = Generator(m).predict(dcs, gcs, qs)
    save(m, dcs, gcs, qs, preds)
    
