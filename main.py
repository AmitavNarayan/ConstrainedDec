import argparse
import subprocess
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["strategy", "aime", "medqa", "finqa"])
    parser.add_argument("--model-stages", nargs="+", default=["SFT", "DPO", "Instruct"])
    parser.add_argument("--model-names", nargs="+", default=["olmo", "tulu"])
    args = parser.parse_args()

    
    for x in args.datasets:
        for y in args.model_stages:
            for z in args.model_names:
                subprocess.run(["uv", 
                                "run", 
                                "--env-file", 
                                ".env", 
                                "python", 
                                "-m", 
                                "constrained_dec",
                                "--dataset", 
                                x,
                                "--model-stage",
                                y,
                                "--model-name",
                                z
                                ])