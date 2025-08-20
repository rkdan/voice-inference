import click
import os

# this needs to be somewhere, but I have no idea where...so it's everywhere...
# also don't know why models doesn't work, but workspace/models does...
os.environ['HF_HOME'] = 'workspace/models'
os.environ['TRANSFORMERS_CACHE'] = 'workspace/models'

from loguru import logger
from voice_inference.logging import setup_logging
from voice_inference.config import load_config
from voice_inference.infer import VLLMInference, load_questions
from pathlib import Path
from datetime import datetime
import json


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--temp", type=float, help="Override temperature for sampling")
@click.option("--max-tokens", type=int, help="Override max tokens for generation")
@click.option("--n", type=float, help="Override number of samples to generate")
@click.option("--n-gpus", type=int, default=1, help="Number of GPUs to use for inference")
def main(config_path, log_level, log_file, temp, max_tokens, n, n_gpus):
    """Run stylometric analysis"""

    setup_logging(level=log_level, log_file=log_file)

    # == Load config ===
    try:
        logger.info("Loading config from {}", config_path)
        config = load_config(config_path)
        
        # Override sampling parameters if provided via CLI
        if temp is not None:
            config.sampling_params.temperature = temp
            logger.info("Overriding temperature to: {}", temp)
        if max_tokens is not None:
            config.sampling_params.max_tokens = max_tokens
            logger.info("Overriding max_tokens to: {}", max_tokens)
        if n is not None:
            config.sampling_params.n = n
            logger.info("Overriding n to: {}", n)
        if n_gpus is not None:
            config.gpus = n_gpus
            logger.info("Overriding number of GPUs to: {}", n_gpus)
        
        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")

    except Exception as e:
        logger.error("Failed to load config: ", e)
        raise

    # set environment variable for Hugging Face token
    os.environ['HF_TOKEN'] = config.hf_token
    # not sure if this is the right place for these...
    os.environ['HF_HOME'] = 'workspace/models'
    os.environ['TRANSFORMERS_CACHE'] = 'workspace/models'

    # == Run inference ===
    llm = VLLMInference(
        model_name=config.model_name,
        tokenizer_name=config.model_name,
        result_path=config.output_path,
        gpus=config.gpus,
        sampling_params=config.sampling_params  # Pass sampling params to inference
    )

    logger.info("Starting inference with model: {}", config.model_name)

    # Load questions from the input file
    message_list = load_questions(config.input_path)
    logger.info("Loaded {} question pairs from {}", len(message_list), config.input_path)

    # Perform batch inference
    outputs = llm.batch_inference(message_list)

    logger.info("Saving results to {}", config.output_path)
    path = Path(config.output_path)
    experiment_name = config.model_name.replace("/", "_").lower()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Include temperature in output filename for easier identification
    temp_suffix = f"_temp{config.sampling_params.temperature}"
    n_suffix = f"_n{config.sampling_params.n}" if config.sampling_params.n > 1 else ""
    output_path = path / f"{experiment_name}{temp_suffix}{n_suffix}_{timestamp}"
    
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(config.sampling_params.n):
        # we need to create a separate results_i file for each sample
        output_file = output_path / f"results_{i}.json"
        logger.info("Saving results to {}", output_file)
        with open(output_file, 'w') as f:
            json.dump([{'gen_response': output.outputs[i].text} for output in outputs], f, indent=2)


    logger.success("Inference completed successfully! Results saved to {}", output_path)