from configs import *
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class Generator:
    def __init__(self, m: ModelConfig):

        self.llm = LLM(m.path, max_model_len=m.max_model_len,
                       gpu_memory_utilization=m.gpu_mem_util,
                        tensor_parallel_size=m.num_gpus)

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

