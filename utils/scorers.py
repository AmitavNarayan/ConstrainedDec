
import evaluate
import re
import parse

class Scorer():
        def __init__(self):
                self.rouge_method = evaluate.load("rouge")
                self.check = parse.compile("Reason: {reason}\nAnswer: {answer:n}", case_sensitive=True)
        def score(self, ans: str, gold: str, gold_type: str):
                score = 0
                if gold_type=="str":
                        rouge_inputs = {"predictions":[ans], "references":[gold], "rouge_types":["rouge2"], "use_stemmer":True, "use_aggregator":False}
                        score = self.rouge_method.compute(**rouge_inputs)["rouge2"][0]
                if gold_type=="int":
                        nums = re.findall(r"-?\d+\.?\d*", ans.replace(",", ""))
                        if len(nums)>0: score = int(float(nums[-1]) == float(gold))
                return score
        def check_in_format(self, ans: str):
                if self.check.parse(ans.strip()) is not None: return 1
                return 0

