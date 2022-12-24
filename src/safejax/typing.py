# Reference from https://github.com/huggingface/datasets/blob/main/src/datasets/utils/typing.py
import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path, os.PathLike]
