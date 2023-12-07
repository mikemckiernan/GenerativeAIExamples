"""The script used for fetching models from an NGC repository."""
import base64
import glob
import json
import os
import pathlib
import shutil
import sys
import typing
import zipfile

import requests  # type: ignore
from common import mk_model_archive_path, mk_model_cache_dir

# INPUTS
MODEL = os.environ.get("MODEL", "")
MODEL_K8S_NAME = os.environ.get("MODEL_K8S_NAME", "")
NGC_KEY = os.environ.get("NGC_CLI_API_KEY", "")
NGC_ORG = os.environ.get("NGC_CLI_ORG", "")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
MODEL_K8S_NAME = os.environ.get("MODEL_K8S_NAME", "")
NUM_GPUS = os.environ.get("NUM_GPUS", "0")

# CONSTANTS
NGC_AUTH_URL = "https://authn.nvidia.com/token"
MODEL_REPO = "/models"
MODEL_CACHE = "/models/cache"


def parse_model_full_name(
    model_name: str,
) -> typing.Tuple[str, typing.Optional[str], str, str]:
    """
    Parse model parts.

    Find the org, team, model name, and model version for the requested model
    to download from the model repository.

    Parameters
    ----------
    model_name: str - A string of the model name to download from the repository.

    Returns
    -------
    Tuple: Returns a Tuple of org name, team name (if applicable), model name, and model version.
    """
    model_version_parts = model_name.split(":")
    if len(model_version_parts) < 2:
        raise ValueError(f"Invalid model name, no version provided: {model_name}")
    version = model_version_parts[1]

    model_parts = model_version_parts[0].split("/")
    if len(model_parts) < 2 or len(model_parts) > 3:
        raise ValueError(f"Invalid model name, can't find org and model: {model_name}")

    if len(model_parts) < 3:
        return (model_parts[0], None, model_parts[1], version)
    return (model_parts[0], model_parts[1], model_parts[2], version)


def get_ngc_auth_header(org_name: str, api_key: str) -> typing.Dict[str, str]:
    """
    Build the NGC authentication header.

    Parameters
    ----------
    org_name: str - The string of the org name where the model is located.
    api_key: str - The user's API key for the model repository.

    Returns
    -------
    Dict: Returns a dictionary of the authenticated NGC headers with the bearer token.
    """
    ngc_scope = f"group/ngc:{org_name}"
    query_string = {"scope": ngc_scope}
    auth_string = f"$oauthtoken:{api_key}"

    # Generate authorisation based on your API Key (standard base64 encoding)
    encoded_bytes = base64.b64encode(auth_string.encode("utf-8"))
    encoded_auth_str = str(encoded_bytes, "utf-8")

    # Login to NGC API
    headers = {
        "Authorization": f"Basic {encoded_auth_str}",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Host": "authn.nvidia.com",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "cache-control": "no-cache",
    }
    response = requests.request(
        "GET", NGC_AUTH_URL, headers=headers, params=query_string, timeout=60
    )
    json_response = json.loads(response.text)

    return {
        "Authorization": f"Bearer {json_response['token']}",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "cache-control": "no-cache",
    }


def create_model_base_url(org: str, team: typing.Optional[str], model: str) -> str:
    """
    Build the base model URL for the request.

    Parameters
    ----------
    org: str - The org name where the model is stored.
    team: str - The team name if applicable.

    Returns
    -------
    str: A string of the final URL for requests.
    """
    if team:
        return f"https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/models/{model}/"
    return f"https://api.ngc.nvidia.com/v2/org/{org}/models/{model}/"


def print_progress(fetched: float, last_fetched: float) -> float:
    """
    Print the latest progress downloading the file.

    Parameters
    ----------
    fetched: float - The number of bytes that have been downloaded.
    last_fetched: float - The number of GB that were downloaded in the last pass.

    Returns
    -------
    Float: Returns a float of the number of GB that were downloaded during this pass.
    """
    fetched_gb = fetched / 1024.0 / 1024.0 / 1024.0
    if fetched_gb > last_fetched + 0.5:
        sys.stdout.write(f"Downloaded {fetched_gb:0.2f}GB\n")
        last_fetched = fetched_gb
    return last_fetched


# pylint: disable-next=too-many-arguments
def download_model(
    headers: typing.Dict[str, str],
    org: str,
    team: typing.Optional[str],
    model: str,
    version: str,
    local: str,
) -> None:
    """
    Download the model from the model repository.

    Parameters
    ----------
    headers: Dict - A dictionary of the authentication headers.
    org: str - The name of the org where the model is stored.
    team: str - The team name if applicable.
    model: str - The name of the model to download.
    version: str - The version of the model to download.
    local: str - Path where the model should be saved locally.

    Returns
    -------
    str: Returns the name of the downloaded file.
    """
    # determine remote url
    base_url = create_model_base_url(org, team, model)
    url = f"{base_url}/versions/{version}/zip"

    if pathlib.Path(local).is_file():
        # os.remove(local)
        return

    # download model
    fetched = 0
    last_fetched_msg = print_progress(fetched, -2)
    with requests.get(url, headers=headers, stream=True, timeout=300) as req:
        req.raise_for_status()
        with open(local, "wb") as out:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    fetched += 8192
                    last_fetched_msg = print_progress(fetched, last_fetched_msg)
                    out.write(chunk)
    _ = print_progress(fetched, -2)


def extract_model(
    org: str,
    team: typing.Optional[str],
    model: str,
    version: str,
    fname: str,
    output_dir: str,
) -> None:
    """
    Extract the downloaded model locally.

    Parameters
    ----------
    org: str - The name of the org where the model is stored.
    team: str - The team name if applicable.
    model: str - The name of the model to download.
    version: str - The version of the model to download.
    fname: str - The name of the downloaded file.
    output_dir: str - The directory to extract to.
    """
    if team:
        model_full_name = f"{org}_{team}_{model}"
    else:
        model_full_name = f"{org}_{model}"
    sys.stdout.write(f"Extracting model to {output_dir}...\n")

    # extract the zip file
    with zipfile.ZipFile(fname, "r") as zipf:
        zipf.extractall(output_dir)

    # move the contents from the NGC model name directory to the cache directory
    for fname in glob.glob(os.path.join(output_dir, model_full_name, "*")):
        sys.stdout.write(os.path.join(output_dir, fname))
        shutil.move(fname, output_dir)


def main() -> None:
    """Execute the main routine for this script."""
    org, team, model, version = parse_model_full_name(MODEL)

    # construct file paths
    cache_dir = mk_model_cache_dir(
        model_name=MODEL_K8S_NAME,
        version=MODEL_VERSION,
    )
    zip_file = mk_model_archive_path(
        model_name=MODEL_K8S_NAME,
        version=MODEL_VERSION,
    )

    # ensure the model doesn't already exist
    if os.path.exists(cache_dir) and os.listdir(cache_dir):
        sys.stdout.write("Extracted model already found in cache.")
        return

    # download model from ngc repo
    if os.path.exists(zip_file):
        sys.stdout.write("Model archive already downloaded...\n\n")
    else:
        sys.stdout.write("Downloading model...\n")
        auth_headers = get_ngc_auth_header(NGC_ORG, NGC_KEY)
        download_model(auth_headers, org, team, model, version, zip_file)

    # extract the model zip
    sys.stdout.write("\n\nExtracting model...\n")
    extract_model(org, team, model, version, zip_file, cache_dir)
    os.remove(zip_file)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
