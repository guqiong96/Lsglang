"""Environment-variable primitives (leaf module: no sglang imports)."""

import logging
import os

logger = logging.getLogger(__name__)

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        # Warn once per env var key (not per value), otherwise different keys that share the
        # same invalid value may suppress warnings incorrectly.
        if name not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) encountered unrecognized value={value} and will treat as false"
            )
        _warned_bool_env_var_keys.add(name)

    return value in truthy_values


def get_int_env_var(name: str, default: int = 0) -> int:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default
    
def set_int_env_var(name: str, value: int):
    os.environ[name] = str(value)


def get_str_env_var(var_name: str, default: str | None = None) -> str:
    val = os.getenv(var_name)
    if val is None:
        val = ""
    if default is None:
        default = ""
    val = val.strip()
    return val or default
    
