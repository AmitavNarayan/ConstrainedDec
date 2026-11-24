FMT = """<format>
Format your response exactly as follows:
Reason: <your reasoning>
Answer: <your answer>
</format>
"""


TASK = """<task>
Think through the following question, step by step. 
Your answer should be {text_fmt}.
</task>
<question>
{question}
</question>
"""

REWRITE = """Rewrite the following response in the required format. Do not change the answer.
<response>
{response}
</response>
"""

JUDGE_MSG = """Given a task and a response, extract the answer from the response.
Do not solve the task, use outside knowledge, or modify the answer in any way. 
<task>
Think through the following question, step by step. 
Your answer should be {text_fmt}.
</task>
<response>
{response}
</response>
"""


ds_fmts = {
    "strategy": {"text_fmt": "a single word: true or false",
                     "regex_fmt": r"true|false"},
    "aime": {"text_fmt": "a single integer from 0 to 999, inclusive",
             "regex_fmt": r"\d{1,3}"},
    "medqa": {"text_fmt": "a single letter: A, B, C, D, or E",
                "regex_fmt": r"[A-E]"},
    "finqa": {"text_fmt": "a single numeric value.",
              "regex_fmt": r"-?\d[\d,]*(\.\d+)?%?"

}



