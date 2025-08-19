import yaml
from pydantic import BaseModel, Field, ValidationError

class InferenceConfig(BaseModel):
    """Configuration for inference"""
    model_name: str = Field(..., description="Name of the model to use")
    gpus: int = Field(1, description="Number of GPUs to use")
    input_path: str = Field(..., description="Path to the input data file")
    output_path: str = Field(..., description="Path to save the output results")
    hf_token: str = Field(..., description="Hugging Face token for model access")


def load_config(config_path: str) -> InferenceConfig:
    """Load and validate configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return InferenceConfig(**config_dict)