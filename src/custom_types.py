""" Custom types for Pydantic models. """
from typing import Optional, Union
import numpy as np


class RoundedFloat(float):
    """Rounds floats to a specified number of decimal places, defaulting to 3."""

    def __new__(cls, value: float, n_decimals: Optional[int] = 3) -> "RoundedFloat":
        if n_decimals is not None:
            cls.n_decimals = n_decimals
        return super().__new__(cls, value)

    @classmethod
    def __get_validators__(cls):  # type: ignore
        """Pydantic validator for RoundedFloat"""
        yield cls.validate

    @classmethod
    def validate(cls, v: Union[float, np.float32, np.float64]) -> "RoundedFloat":
        """Pydantic validator for RoundedFloat"""
        if isinstance(v, (np.float32, np.float64)):
            v = float(v)  # Convert numpy float to standard Python float
        return cls(round(v, cls.n_decimals))
