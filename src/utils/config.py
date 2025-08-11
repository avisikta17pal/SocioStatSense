"""Configuration management for the adaptive modeling framework."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    provider: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    update_frequency: str = "daily"
    

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""
    missing_value_strategy: str = "interpolate"
    outlier_detection: str = "iqr"
    outlier_threshold: float = 3.0
    normalization: str = "standard"
    time_alignment: str = "1H"


class ModelingConfig(BaseModel):
    """Configuration for modeling parameters."""
    window_size: int = 168
    update_frequency: str = "1H"
    change_point_min_size: int = 24
    sparse_selection_alpha: float = 0.01
    uncertainty_quantiles: list = Field(default=[0.05, 0.25, 0.75, 0.95])


class DashboardConfig(BaseModel):
    """Configuration for dashboard settings."""
    refresh_interval: int = 60
    max_display_points: int = 1000
    default_forecast_horizon: int = 72


class AlertsConfig(BaseModel):
    """Configuration for alerts and monitoring."""
    change_point_threshold: float = 0.8
    anomaly_threshold: float = 2.5
    model_drift_threshold: float = 0.1
    notification_methods: list = Field(default=["email", "dashboard"])


class Config(BaseModel):
    """Main configuration class."""
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    
    # Environment variables
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    app_port: int = Field(default=8501)
    database_url: str = Field(default="sqlite:///data/socioeconomic.db")
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "True").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            app_port=int(os.getenv("APP_PORT", "8501")),
            database_url=os.getenv("DATABASE_URL", "sqlite:///data/socioeconomic.db")
        )


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "data_sources.yaml"
    
    # Load YAML configuration
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Create config from environment variables
    config = Config.from_env()
    
    # Update with YAML configuration
    if yaml_config:
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def get_api_key(env_var_name: str) -> Optional[str]:
    """Get API key from environment variables."""
    api_key = os.getenv(env_var_name)
    if not api_key:
        print(f"Warning: {env_var_name} not found in environment variables")
    return api_key