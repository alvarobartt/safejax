try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False

from safetensors.torch import save_file

from safejax.typing import PathLike

if has_torch:

    def from_bin(in_filename: PathLike, out_filename: PathLike) -> PathLike:
        """Convert a bin file to a safetensors file.

        Args:
            in_filename: Path to the bin file.
            out_filename: Path to the safetensors file.

        Returns:
            Path to the safetensors file.
        """
        state_dict = torch.load(in_filename, map_location="cpu")
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(state_dict, out_filename)
        return out_filename
