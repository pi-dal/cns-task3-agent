"""Container runtime: input mapping, output packing, fail-fast."""

import os
import shutil
import zipfile
from pathlib import Path


# Competition contract: output file must be named output.zip
OUTPUT_ZIP_NAME = "output.zip"


def resolve_input_dir(custom_path: str | None = None) -> str:
    """Resolve input directory from argument or default /saisdata."""
    if custom_path:
        p = custom_path
    else:
        p = os.environ.get("INPUT_DIR", "/saisdata")
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Input directory not found: {p}")
    return p


def resolve_output_dir(custom_path: str | None = None) -> str:
    """Resolve output directory from argument or default /saisresult."""
    if custom_path:
        p = custom_path
    else:
        p = os.environ.get("OUTPUT_DIR", "/saisresult")
    os.makedirs(p, exist_ok=True)
    return p


def pack_result(output_dir: str, result_zip: str | None = None) -> str:
    """Pack the output directory into a zip archive.
    
    Args:
        output_dir: Directory whose contents to pack.
        result_zip: Target zip path (default: output_dir/output.zip per competition contract).
    
    Returns:
        Path to the created zip archive.
    """
    if result_zip is None:
        result_zip = os.path.join(output_dir, OUTPUT_ZIP_NAME)
    
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".zip"):
                    continue
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, output_dir)
                zf.write(full, arcname)
    
    return result_zip
