from pathlib import Path
from utils.constants import DATA_DIR


def get_data_path(dataset_name: str, filepath: str) -> Path:
    path = DATA_DIR.joinpath(dataset_name).joinpath(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_gdrive_file(dataset_name: str, file_id: str, filepath: str) -> Path:
    path = get_data_path(dataset_name, filepath)
    if not path.exists():
        download_gdrive_file(file_id, path)
    return path


def get_file(dataset_name: str, url: str, filepath: str, checksum=None) -> Path:
    path = get_data_path(dataset_name, filepath)
    if not path.exists():
        download_file(url, path)
    if checksum:
        check_file_hash(path, checksum)
    return path


def download_gdrive_file(file_id: str, destination: str | Path):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination))


def download_file(url: str, file: Path):
    import tqdm
    from urllib.request import urlretrieve
    pbar = tqdm.tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(file.name))

    def update_hook(block_number, read_size, total_size):
        pbar.total = total_size
        pbar.update(read_size)
    urlretrieve(url, file, reporthook=update_hook)
    pbar.clear()


def check_file_hash(file: Path, hashcode: str):
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file, "rb") as f:
        for byte_block in iter(lambda: f.read(2**16), b""):
            sha256_hash.update(byte_block)
    file_hashcode = sha256_hash.hexdigest()
    assert file_hashcode == hashcode, f"Unexpected content of file {file.name}."


def load_b64(file: Path | str) -> str:
    import base64
    with open(file, "rb") as fd:
        return base64.b64encode(fd.read()).decode("utf-8")
