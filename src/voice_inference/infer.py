import json
import os
import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from voice_inference.config import SamplingConfig


def load_questions(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            pairs.append(json.loads(line))

    message_list = [data['messages'][0:2] for data in pairs]

    return message_list


class VLLMInference:
    def __init__(self, model_name, tokenizer_name, result_path, gpus=1, sampling_params=None):
        os.environ['HF_HOME'] = 'workspace/models'
        os.environ['TRANSFORMERS_CACHE'] = 'workspace/models'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=gpus,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            max_model_len=16384,
        )
        self.result_path = result_path
        self.sampling_params = sampling_params or SamplingConfig()

    def batch_inference(self, message_list):
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to False to strictly disable thinking
            ) for messages in message_list
        ]
        
        vllm_sampling_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            max_tokens=self.sampling_params.max_new_tokens,
            n=self.sampling_params.n,
        )
        outputs = self.model.generate(texts, sampling_params=vllm_sampling_params)

        return outputs
