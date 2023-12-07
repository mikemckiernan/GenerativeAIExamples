"""Script to delete a model from the model repo."""
import os
import shutil

from common import mk_model_output_dir

# INPUTS
MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
MODEL_K8S_NAME = os.environ.get("MODEL_K8S_NAME", "")
NUM_GPUS = os.environ.get("NUM_GPUS", "0")
COMPUTE_CAP_MAJOR = os.environ.get("COMPUTE_CAP_MAJOR", "0")
COMPUTE_CAP_MINOR = os.environ.get("COMPUTE_CAP_MINOR", "0")


def main() -> None:
    """
    Execute the main routine for this script.

    Parameters
    ----------
    model: str - The name of the model
    ngc_key: str - The API key to use to authenticate with NGC
    ngc_org: str - The Organization name used to authenticate with NGC
    """
    num_gpus = int(NUM_GPUS)
    # FUTURE: decide when, if ever, to clean up the model cache
    # construct file paths
    compute_cap = (COMPUTE_CAP_MAJOR, COMPUTE_CAP_MINOR)
    output_dir = mk_model_output_dir(
        model_name=MODEL_K8S_NAME,
        version=MODEL_VERSION,
        num_gpus=num_gpus,
        compute_cap=compute_cap,
        ensure_exists=False,
    )

    # remove model directory
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
