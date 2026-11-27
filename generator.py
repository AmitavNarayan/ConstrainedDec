from configs import *
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class Generator:
    def __init__(self, m: ModelConfig):
        self.llm = LLM(m.model_origin+m.name, max_model_len = m.max_model_len, gpu_memory_utilization=m.gpu_mem_util)
    def generate(self, msg_contents: list[list[str]], gs: list[GenConfig]) -> list[list[str]]:
        params = []
        inputs = []
        for g, mc in zip(gs, msg_contents):
            for c in mc:
                params.append(SamplingParams(
                                temperature=g.temp, 
                                max_tokens = g.max_tokens,
                                structured_outputs = StructuredOutputsParams(regex=g.regex) if g.regex else None))
                inputs.append([{"role": "user", "content": c}]) 
        raw_outputs = [r.outputs[0].text for r in self.llm.chat(inputs, sampling_params=params)]
        outputs = []
        start = 0
        for mc in msg_contents:
            outputs.append(raw_outputs[start: start + len(mc)])
            start += len(mc)
        return outputs

    def judge(self, dcs: list[DataConfig], gcs: list[GenConfig], resps: list[list[str]]) -> tuple:
        msg_contents = [[dc.inst.format(answer_des=dc.answer_des, response=r) for r in res] for dc, res in zip(dcs, resps)]
        judge_outputs = self.generate(msg_contents, gcs)
        accs = [dc.score(o) for (dc, o) in zip(dcs, judge_outputs)]
        return judge_outputs, accs

    def predict(self, dcs: list[DataConfig], gcs: list[GenConfig], ques: list[list[str]]) -> list[list[str]]:
        msg_contents = [[dc.inst.format(answer_des=dc.answer_des, question=q) for q in qs] for dc, qs in zip(dcs, ques)]
        return self.generate(msg_contents, gcs)
    






