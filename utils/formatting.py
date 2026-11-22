INSTRUCTION = """<format>
Format your response exactly as follows:
Reason: {reason}
Answer: {answer}
</format>
<task>
{task_des}
{question}
</task>
"""



ds_formats = {
    "gsm": {
        "answer_format": r"\s*Reason:\s*([\s\S]*?)\s*\nAnswer:\s*(-?[\d,]+)\s*",
        "reason":"<your reasoning>", 
        "answer":"<your answer, an integer>",
        "task_des":"Think through the following question, step by step."
    },
}






