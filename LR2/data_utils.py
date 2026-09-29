from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/166/hill%2Bvalley.zip"
DEFAULT_TRAIN_FILE = "Hill_Valley_without_noise_Training.data"
EXPECTED_FEATURES = [f"X{i}" for i in range(1, 101)]
TARGET = "class"


def _download_uci_zip(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "hill_valley_uci.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path

    request = Request(
        UCI_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Не вдалося автоматично завантажити Hill-Valley з UCI. "
            "Завантажте Hill_Valley_without_noise_Training.data вручну та "
            "передайте шлях через --data-file."
        ) from exc

    zip_path.write_bytes(data)
    return zip_path


def ensure_training_file(data_file: str | None, cache_dir: str | Path = "data") -> Path:
    """Return a local path to the selected Hill-Valley training file."""
    if data_file:
        path = Path(data_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {path}")
        return path

    cache_dir = Path(cache_dir)
    zip_path = _download_uci_zip(cache_dir)
    extracted = cache_dir / DEFAULT_TRAIN_FILE
    if extracted.exists() and extracted.stat().st_size > 0:
        return extracted

    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if Path(name).name == DEFAULT_TRAIN_FILE
        ]
        if not candidates:
            raise FileNotFoundError(
                f"У ZIP UCI не знайдено {DEFAULT_TRAIN_FILE}. "
                f"Доступні файли: {archive.namelist()[:10]}"
            )
        extracted.write_bytes(archive.read(candidates[0]))

    return extracted


def load_hill_valley(data_file: str | None = None, cache_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.Series]:
    """Load Hill-Valley data and validate the expected schema."""
    path = ensure_training_file(data_file, cache_dir=cache_dir)
    df = pd.read_csv(path)

    required = EXPECTED_FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        # Some mirrors may be missing the header. Re-read as headerless data.
        if df.shape[1] == 101:
            df = pd.read_csv(path, header=None, names=required)
        else:
            raise ValueError(f"У файлі відсутні стовпці: {missing}")

    X = df[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Виявлено пропущені або некоректні числові значення.")

    y = y.astype(int)
    unexpected = sorted(set(y.unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"Неприпустимі значення class: {unexpected}")

    return X, y
