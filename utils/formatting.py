EXACT_FMT = """<format>
Format your response exactly as follows:
Reason: {reason}
Answer: {answer}
</format>
"""

APPROX_FMT = """Try your best to format your response in the following style.
Reason: {reason}
Answer: {answer}
"""


TASK = """<task>
{task_des}
{question}
</task>
"""

REWRITE = """
<response>
Rewrite the following response in the required format. Do not change the answer.
{response}
</response>
"""


ds_formats = {
    "gsm": {
        "answer_format": r"\s*Reason:\s*([\s\S]*?)\s*\nAnswer:\s*(-?\d[\d,]*)\s*",
        "reason":"<your reasoning>", 
        "answer":"<your answer, an integer>",
        "task_des":"Think through the following question, step by step."
    },
}

acc = r"-?\d[\d,]*"







