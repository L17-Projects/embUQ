from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union
import os
import yaml
from .models import InferenceConfig, PropagationConfig, SamplingConfig

T = TypeVar('T')


def resolve_inference_config_path(project_root: Union[str, Path], experiment: str = 'compression', mode: str = 'production') -> Path:
    override = os.getenv('HUQ_INFERENCE_CONFIG') or os.getenv('CONFIG_PATH')
    if override:
        candidate = Path(override)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = Path(project_root, override)
        if candidate.exists():
            return candidate
    project_root = Path(project_root)
    config_dir = project_root / 'inference' / 'configs' / mode
    mapping = {
        ('production', 'compression'): ['inference_config_compression.yaml', 'inference_config.yaml'],
        ('production', 'indentation'): ['inference_config_indentation.yaml'],
        ('test', 'compression'): ['inference_config.yaml', 'inference_config_compression.yaml'],
        ('test', 'indentation'): ['inference_config_indentation.yaml'],
    }
    if (mode, experiment) not in mapping:
        raise ValueError(f'Unsupported inference config selection: mode={mode}, experiment={experiment}')
    for name in mapping[(mode, experiment)]:
        candidate = config_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not find inference config under {config_dir}')


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def load_config(path: Union[str, Path], model_class: Type[T], strict: bool = True) -> T:
    return model_class.model_validate(load_yaml(path))


def load_inference_config(path: Union[str, Path], strict: bool = True) -> InferenceConfig:
    return load_config(path, InferenceConfig, strict=strict)


def load_sampling_config(path: Union[str, Path], strict: bool = True) -> SamplingConfig:
    return load_config(path, SamplingConfig, strict=strict)


def load_propagation_config(path: Union[str, Path], strict: bool = True) -> PropagationConfig:
    return load_config(path, PropagationConfig, strict=strict)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_inference_config_with_overrides(base_path: Union[str, Path], override_path: Optional[Union[str, Path]] = None, cli_overrides: Optional[Dict[str, Any]] = None) -> InferenceConfig:
    data = load_yaml(base_path)
    if override_path is not None:
        data = merge_configs(data, load_yaml(override_path))
    if cli_overrides is not None:
        data = merge_configs(data, cli_overrides)
    return InferenceConfig.model_validate(data)


def validate_config_file(path: Union[str, Path], config_type: str = 'inference') -> Tuple[bool, str]:
    model_map = {'inference': InferenceConfig, 'sampling': SamplingConfig, 'propagation': PropagationConfig}
    if config_type not in model_map:
        return False, f'Unknown config type: {config_type}. Valid: {list(model_map.keys())}'
    try:
        load_config(path, model_map[config_type])
        return True, ''
    except FileNotFoundError as e:
        return False, f'File not found: {e}'
    except yaml.YAMLError as e:
        return False, f'YAML parsing error: {e}'
    except Exception as e:
        return False, f'Validation error: {e}'
