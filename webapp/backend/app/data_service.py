from __future__ import annotations

from pathlib import Path
import uuid
import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

SUPPORTED_UPLOAD_SUFFIXES = {".csv"}
REQUIRED_WAVEFORM_COLUMNS = {"wave_id", "sample", "time_ms", "value"}
WIDE_SIGNAL_COLUMNS = {"Signal"}


def save_uploaded_file(file: FileStorage, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file.filename or "")
    if not original_name:
        raise ValueError("Missing upload filename")

    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))
        raise ValueError(f"Unsupported file type '{suffix}'. Upload a {supported} file.")

    filename = f"{Path(original_name).stem[:80]}_{uuid.uuid4().hex[:10]}{suffix}"
    path = (upload_dir / filename).resolve()
    upload_root = upload_dir.resolve()
    if upload_root not in path.parents:
        raise ValueError("Invalid upload path")

    file.save(path)
    return path


def read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    supported = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))
    raise ValueError(f"Unsupported file type '{suffix}'. Supported files: {supported}")


def load_dataset(path: Path) -> pd.DataFrame:
    return read_tabular_file(path)


def build_preview(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    missing = df.isna().sum().sort_values(ascending=False).head(20)

    wave_count = int(df["wave_id"].nunique()) if "wave_id" in df.columns else 0

    # round numeric columns ให้เหลือ 2 ตำแหน่งใน preview
    preview_df = df.head(20).copy()
    for col in numeric_columns:
        if col in preview_df.columns:
            preview_df[col] = preview_df[col].round(2)

    sample_count = 0
    if "wave_id" in df.columns and "sample" in df.columns:
        first_wave = df["wave_id"].iloc[0]
        sample_count = int((df["wave_id"] == first_wave).sum())

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "preview": preview_df.fillna("").to_dict(orient="records"),
        "numeric_columns": numeric_columns,
        "wave_count": wave_count,    # เพิ่ม
        "sample_count": sample_count,
        "missing_top20": [
            {"column": str(col), "missing": int(val)}
            for col, val in missing.items()
        ],
    }


def build_preview_from_csv(path: Path, chunksize: int = 200_000) -> dict:
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV is supported for now")

    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    first_chunk = next(reader, None)
    if first_chunk is None:
        raise ValueError("CSV is empty")

    columns = first_chunk.columns.tolist()
    numeric_columns = first_chunk.select_dtypes(include="number").columns.tolist()
    preview_df = first_chunk.head(20).copy()

    row_count = 0
    missing_counts = pd.Series(0, index=columns, dtype="int64")
    wave_ids = set()
    first_wave = None
    sample_count = 0

    def consume(chunk: pd.DataFrame) -> None:
        nonlocal row_count, missing_counts, first_wave, sample_count
        row_count += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")

        if "wave_id" in chunk.columns:
            wave_ids.update(chunk["wave_id"].dropna().astype(str).unique().tolist())
            if first_wave is None and not chunk.empty:
                first_wave = chunk["wave_id"].iloc[0]
        if first_wave is not None and "wave_id" in chunk.columns and "sample" in chunk.columns:
            sample_count += int((chunk["wave_id"] == first_wave).sum())

    consume(first_chunk)
    for chunk in reader:
        consume(chunk)

    for col in numeric_columns:
        if col in preview_df.columns:
            preview_df[col] = preview_df[col].round(2)

    missing_top20 = missing_counts.sort_values(ascending=False).head(20)

    return {
        "shape": [int(row_count), int(len(columns))],
        "columns": columns,
        "preview": preview_df.fillna("").to_dict(orient="records"),
        "numeric_columns": numeric_columns,
        "wave_count": int(len(wave_ids)),
        "sample_count": int(sample_count),
        "missing_top20": [
            {"column": str(col), "missing": int(val)}
            for col, val in missing_top20.items()
        ],
    }


def build_preview_from_file(path: Path, chunksize: int = 200_000) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        preview = build_preview_from_csv(path, chunksize=chunksize)
        validate_waveform_columns(preview["columns"])
        return preview
    raise ValueError(f"Unsupported file type '{suffix}'. Upload a CSV file.")


def validate_waveform_columns(columns: list[str]) -> None:
    column_set = set(columns)
    has_long_schema = REQUIRED_WAVEFORM_COLUMNS.issubset(column_set)
    has_wide_schema = WIDE_SIGNAL_COLUMNS.issubset(column_set) and any(str(col).endswith(":") for col in columns)
    if has_long_schema or has_wide_schema:
        return

    required = ", ".join(sorted(REQUIRED_WAVEFORM_COLUMNS))
    raise ValueError(
        "Invalid waveform CSV schema. Use long format columns "
        f"({required}) or wide signal format with Signal plus columns ending in ':'."
    )
