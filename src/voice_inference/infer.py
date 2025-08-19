import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ['HF_HOME'] = 'models'
os.environ['TRANSFORMERS_CACHE'] = 'models'

def load_questions(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            pairs.append(json.loads(line))

    message_list = [data['messages'][0:2] for data in pairs]

    return message_list


class VLLMInference:
    def __init__(self, model_name, tokenizer_name, result_path, gpus=1):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = LLM(model=model_name, tensor_parallel_size=gpus)
        self.result_path = result_path

    def batch_inference(self, message_list):
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to False to strictly disable thinking
            ) for messages in message_list
        ]
        
        sampling_params = SamplingParams(max_tokens=2048)
        outputs = self.model.generate(texts, sampling_params=sampling_params)

        self._save_outputs(outputs)

    def _save_outputs(self, outputs):

        generated_responses = [{'gen_response' : output.outputs[0].text} for output in outputs]

        with open(self.result_path, 'w') as f:
            json.dump(generated_responses, f, indent=2)