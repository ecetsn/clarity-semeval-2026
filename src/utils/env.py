import os
from pathlib import Path
from typing import Dict, Iterable

from dotenv import load_dotenv


def load_and_validate_env(required_keys: Iterable[str]) -> Dict[str, str]:
    """
    Load environment variables from a local .env file (if present) and ensure
    the provided keys exist. Returns a mapping of the resolved variables.
    """
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback to system environment
        load_dotenv()

    resolved = {}
    missing = []
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            missing.append(key)
        else:
            resolved[key] = value

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Create a .env file (see .env.example) or export them in your shell."
        )

    # Provide sensible defaults for optional OpenRouter parameters.
    resolved.setdefault(
        "OPENROUTER_API_URL",
        os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/embeddings"),
    )
    return resolved

