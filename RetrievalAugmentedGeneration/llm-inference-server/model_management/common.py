"""Common routines available to all fetchers, converters, and cleaners."""
import os
import pathlib
import typing

# CONSTANTS
MODEL_REPO = "/models"


def mk_model_archive_path(
    model_name: str,
    version: str,
    model_dir: typing.Optional[str] = None,
) -> str:
    """Construct model directory paths and ensure they exist."""
    if not model_dir:
        model_dir = MODEL_REPO

    fname = f"{model_name}_{version}.zip"
    path = os.path.join(model_dir, fname)

    return path


def mk_model_cache_dir(
    model_name: str,
    version: str,
    model_dir: typing.Optional[str] = None,
    ensure_exists: bool = True,
) -> str:
    """Construct model directory paths and ensure they exist."""
    if not model_dir:
        model_dir = MODEL_REPO

    directory = os.path.join(model_dir, model_name, version)
    if ensure_exists:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    return directory


def mk_model_output_dir(
    model_name: str,
    version: str,
    num_gpus: int,
    compute_cap: typing.Tuple[str, str],
    repo_dir: typing.Optional[str] = None,
    ensure_exists: bool = True,
) -> str:
    """Construct model directory paths and ensure they exist."""
    if not repo_dir:
        repo_dir = MODEL_REPO

    directory = os.path.join(
        repo_dir,
        f"{model_name}--vers{version}-shards{num_gpus}-computecap{'.'.join(compute_cap)}",
        "1",
    )
    if ensure_exists:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    return directory
