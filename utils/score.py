

import re
from utils.formatting import ds_formats
def score(ans: str, gold: int, ds: str = "gsm"):
        result = re.fullmatch(ds_formats[ds]["answer_format"], ans.strip())
        res = 0
        if result: 
                try: 
                        if float(gold) == float(result.group(2).replace(",","")): res = 1
                except ValueError: 
                        pass
        return res

