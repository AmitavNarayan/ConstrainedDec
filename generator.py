from configs import *
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class Generator:
    def __init__(self, m: ModelConfig):

        self.llm = LLM(m.path, max_model_len=m.max_model_len,
                       gpu_memory_utilization=m.gpu_mem_util)

    def generate(self, msg_contents: list[list[str]], gcs: list[GenConfig]):
        inputs = [[[{"role": "user", "content": c}] for c in mc] for mc in msg_contents]
        params = [SamplingParams(temperature=g.temp, max_tokens=g.max_tokens,
                                structured_outputs=StructuredOutputsParams(regex=g.regex) if g.regex else None)
                for g in gcs]
        flat_inputs = [msg for group in inputs for msg in group]
        flat_params = [p for p, mc in zip(params, msg_contents, strict=True) for _ in mc]
        raw_outputs = [r.outputs[0].text for r in self.llm.chat(flat_inputs, sampling_params=flat_params)]
        outputs, start = [], 0
        for mc in msg_contents:
            outputs.append(raw_outputs[start:start + len(mc)])
            start += len(mc)
        return inputs, outputs

    def judge(self, dcs, gcs, resps):
        msg_contents = [[dc.inst.format(answer_des=dc.answer_des, response=r) for r in res]
                        for dc, res in zip(dcs, resps, strict=True)]
        inputs, outputs = self.generate(msg_contents, gcs)
        accs = [dc.score(o) for dc, o in zip(dcs, outputs, strict=True)]
        return inputs, outputs, accs

    def predict(self, dcs, gcs, ques):
        msg_contents = [[dc.inst.format(answer_des=dc.answer_des, question=q) for q in qs]
                        for dc, qs in zip(dcs, ques, strict=True)]
        return self.generate(msg_contents, gcs)