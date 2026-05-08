from __future__ import annotations

from pathlib import Path
import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


def save_uploaded_file(file: FileStorage, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename or "uploaded.csv")
    path = upload_dir / filename
    file.save(path)
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV is supported for now")
    return pd.read_csv(path)


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