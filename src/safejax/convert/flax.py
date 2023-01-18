from flax.serialization import msgpack_restore

from safejax.core.save import serialize
from safejax.typing import PathLike


def from_msgpack(in_filename: PathLike, out_filename: PathLike) -> PathLike:
    """Convert a msgpack file to a safetensors file.

    Args:
        in_filename: Path to the msgpack file.
        out_filename: Path to the safetensors file.

    Returns:
        Path to the safetensors file.
    """
    with open(in_filename, mode="rb") as f:
        msgpack_bytes = f.read()
        params = msgpack_restore(msgpack_bytes)
        out_filename = serialize(params, filename=out_filename)
    return out_filename
