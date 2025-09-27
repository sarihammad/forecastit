"""Random seed management for reproducibility."""

import random

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def set_random_seeds(
    random_seed: int = 42,
    numpy_seed: int | None = None,
    python_seed: int | None = None,
) -> None:
    """Set random seeds for reproducibility.

    Args:
        random_seed: Base seed for all random number generators.
        numpy_seed: Specific seed for NumPy (defaults to random_seed).
        python_seed: Specific seed for Python random (defaults to random_seed).
    """
    if numpy_seed is None:
        numpy_seed = random_seed
    if python_seed is None:
        python_seed = random_seed

    # Set Python random seed
    random.seed(python_seed)

    # Set NumPy random seed
    np.random.seed(numpy_seed)

    logger.info(
        "Random seeds set",
        random_seed=random_seed,
        numpy_seed=numpy_seed,
        python_seed=python_seed,
    )


def get_deterministic_hash(text: str, seed: int = 42) -> int:
    """Generate a deterministic hash from text using a seed.

    Args:
        text: Text to hash.
        seed: Seed for hash generation.

    Returns:
        Deterministic hash value.
    """
    # Use Python's built-in hash with a seed
    random.seed(seed)
    hash_value = hash(text)
    return abs(hash_value)


def set_seed_from_string(text: str, base_seed: int = 42) -> int:
    """Set random seed based on a string (useful for per-series seeds).

    Args:
        text: String to derive seed from.
        base_seed: Base seed value.

    Returns:
        Derived seed value.
    """
    derived_seed = get_deterministic_hash(text, base_seed)
    set_random_seeds(derived_seed)
    return derived_seed
