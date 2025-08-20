import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path

class SamplingConfig(BaseModel):
    """Configuration for sampling"""
    temperature: float | list[float] = Field(1.0, description="Sampling temperature or list of temperatures")
    n: int = Field(1, description="Number of samples to generate")
    max_new_tokens: int = Field(2048, description="Maximum number of new tokens")

class InferenceConfig(BaseModel):
    """Configuration for inference"""
    model_name: str = Field(..., description="Name of the model to use")
    gpus: int = Field(1, description="Number of GPUs to use")
    input_path: str = Field(..., description="Path to the input data file")
    output_path: str = Field(..., description="Path to save the output results")
    hf_token: str = Field(..., description="Hugging Face token for model access")
    sampling_params: SamplingConfig = Field(
        default_factory=SamplingConfig, description="Sampling parameters for inference"
    )
    @field_validator("output_path")
    def create_output_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


def load_config(config_path: str) -> InferenceConfig:
    """Load and validate configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return InferenceConfig(**config_dict)