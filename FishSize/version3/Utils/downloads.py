import re
import requests
import shutil
import subprocess
import torch

from pathlib import Path
from urllib import request

from Utils import is_online, TQDM
from Utils.ops import clean_url, url2file


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Example:
        ```python
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file("path/to/file.zip")
        ```
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        # Decide to unzip directly or unzip into a directory
        unzip_as_dir = len(top_level_dirs) == 1  # (len(files) > 1 and not files[0].endswith("/"))
        if unzip_as_dir:
            # Zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # i.e. extract coco8/ dir to ../datasets/
        else:
            # Zip has multiple files at top level
            path = extract_path = Path(path) / Path(file).stem  # i.e. extract multiple files to ../datasets/coco8/

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            return path

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):
            # Ensure the file is within the extract_path to avoid path traversal security vulnerability
            if ".." in Path(f).parts:
                continue
            zipObj.extract(f, extract_path)

    return path


def check_disk_space(url="https://ultralytics.com/assets/coco8.zip", path=Path.cwd(), sf=1.5, hard=True):
    try:
        r = requests.head(url)
        assert r.status_code < 400, f"URL error for {url}: {r.status_code} {r.reason}"
    except Exception:
        return True

    gib = 1 << 30
    data = int(r.headers.get("Content-Length", 0)) / gib
    total, used, free = (x / gib for x in shutil.disk_usage(path))

    if data * sf < free:
        return True

    text = (
        f"WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, "
        f"Please free {data * sf - free:.1f} GB additional disk space and try again."
    )
    if hard:
        raise MemoryError(text)
    return False


def get_google_drive_file_info(link):
    """
    Retrieves the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        (str): Direct download URL for the Google Drive file.
        (str): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Example:
        ```python
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
        ```
    """
    file_id = link.split("/d/")[1].split("/view")[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None

    # Start session
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                f"Google Drive file download quota exceeded. "
                f"Please try again later or download this file manually at {link}."
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"  # v is token
        cd = response.headers.get("content-disposition")
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.

    Example:
        ```python
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```
    """
    gdrive = url.startswith("https://drive.google.com/")  # check if the URL is a Google Drive link
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = Path(dir or ".") / (file or url2file(url))  # URL converted to filename
    if "://" not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        uri = (url if gdrive else clean_url(url)).replace(  # cleaned and aliased url
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "https://ultralytics.com/assets/",  # assets alias
        )
        desc = f"Downloading {uri} to '{f}'"
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        check_disk_space(url, path=f.parent)
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = "sS" * (not progress)  # silent
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
                    assert r == 0, f"Curl return value {r}"
                else:  # urllib download
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(f"Download failure for {uri}. Environment is not online.") from e
                elif i >= retry:
                    raise ConnectionError(f"❌  Download failure for {uri}. Retry limit reached.") from e

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # unzip to dir if provided else unzip in place
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # unzip
        elif f.suffix in {".tar", ".gz"}:
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)
        if delete:
            f.unlink()  # remove zip
        return unzip_dir
