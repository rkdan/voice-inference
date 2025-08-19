import click
import os
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
def main(config_path, log_level, log_file):
    """Run stylometric analysis"""

    setup_logging(level=log_level, log_file=log_file)

    # == Load config ===
    try:
        logger.info("Loading config from {}", config_path)
        config = load_config(config_path)
        logger.success("Config loaded successfully!")
        logger.info("Model: {}", config.model_name)
        logger.info("GPU count: {}", config.gpus)
        logger.info("Data path: {}", config.input_path)
        logger.info("Output path: {}", config.output_path)
        print("")

    except Exception as e:
        logger.error("Failed to load config: ", e)
        raise

    # set environment variable for Hugging Face token
    os.environ['HF_TOKEN'] = config.hf_token

    # == Run inference ===
    llm = VLLMInference(
        model_name=config.model_name,
        tokenizer_name=config.model_name,
        result_path=config.output_path,
        gpus=config.gpus
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
    output_path = path / f"{experiment_name}_{timestamp}"

    generated_responses = [{'gen_response' : output.outputs[0].text} for output in outputs]

    
    with open(output_path, 'w') as f:
        json.dump(generated_responses, f, indent=2)

    logger.success("Inference completed successfully! Results saved to {}", output_path)
