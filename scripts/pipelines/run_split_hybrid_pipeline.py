from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SCRIPT_PATHS = {
    "split_csv_train_test": SCRIPTS / "data" / "split_csv_train_test.py",
    "extract_features": SCRIPTS / "features" / "extract_features.py",
    "make_wave_tensor": SCRIPTS / "data" / "make_wave_tensor.py",
    "train_tcn_encoder": SCRIPTS / "tcn" / "train_tcn_encoder.py",
    "export_tcn_encoder": SCRIPTS / "tcn" / "export_tcn_encoder.py",
    "merge_features_and_embeddings": SCRIPTS / "features" / "merge_features_and_embeddings.py",
    "train_ag_1stage": SCRIPTS / "autogluon" / "train_ag_1stage.py",
    "analyze_regression_preds": SCRIPTS / "analysis" / "analyze_regression_preds.py",
    "features": SCRIPTS / "analysis" / "features.py",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run clean split-before-TCN hybrid waveform pipeline")
    ap.add_argument("--raw", required=True, help="Raw waveform CSV")
    ap.add_argument("--workdir", default="data/split_pipeline")
    ap.add_argument("--tcn-name", default="split_tcn_v1")
    ap.add_argument("--ag-name", default="split_ag_v1")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-len", type=int, default=1000)
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--fast-ms", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse existing outputs and continue missing steps")
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_if_missing(outputs: list[Path], cmd: list[str], resume: bool) -> None:
    if resume and outputs and all(path.exists() for path in outputs):
        joined = ", ".join(str(path) for path in outputs)
        print(f"\n[resume] skip existing: {joined}")
        return
    run(cmd)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    workdir = ROOT / args.workdir
    splits_dir = workdir / "splits"
    processed_dir = workdir / "processed"
    tcn_dir = ROOT / "models" / "TCNModels" / args.tcn_name
    ag_dir = ROOT / "models" / "AutogluonModels" / args.ag_name
    analysis_dir = ROOT / "analysis" / args.ag_name

    if args.resume:
        workdir.mkdir(parents=True, exist_ok=True)
        tcn_dir.mkdir(parents=True, exist_ok=True)
        ag_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output(workdir, args.overwrite)
        prepare_output(tcn_dir, args.overwrite)
        prepare_output(ag_dir, args.overwrite)
        prepare_output(analysis_dir, args.overwrite)
    splits_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    run_if_missing([
        splits_dir / "train.csv",
        splits_dir / "valid.csv",
        splits_dir / "test.csv",
        splits_dir / "split_manifest.csv",
    ], [
        py, str(SCRIPT_PATHS["split_csv_train_test"]),
        "--in", args.raw,
        "--outdir", str(splits_dir),
        "--train-frac", str(args.train_frac),
        "--valid-frac", str(args.valid_frac),
        "--test-frac", str(args.test_frac),
        "--seed", str(args.seed),
    ], args.resume)

    split_csv = {
        "train": splits_dir / "train.csv",
        "valid": splits_dir / "valid.csv",
        "test": splits_dir / "test.csv",
    }

    feature_csv = {}
    tensor_npz = {}
    embed_csv = {}
    hybrid_csv = {}

    for split_name, csv_path in split_csv.items():
        feature_csv[split_name] = processed_dir / f"{split_name}_features.csv"
        tensor_npz[split_name] = processed_dir / f"{split_name}_wave_tensor.npz"
        embed_csv[split_name] = processed_dir / f"{split_name}_tcn_embed.csv"
        hybrid_csv[split_name] = processed_dir / f"{split_name}_hybrid.csv"

        run_if_missing([feature_csv[split_name]], [
            py, str(SCRIPT_PATHS["extract_features"]),
            "--mode", "train",
            "--in", str(csv_path),
            "--out", str(feature_csv[split_name]),
        ], args.resume)
        run_if_missing([tensor_npz[split_name]], [
            py, str(SCRIPT_PATHS["make_wave_tensor"]),
            "--in", str(csv_path),
            "--out", str(tensor_npz[split_name]),
            "--target-len", str(args.target_len),
            "--label-col", args.label,
        ], args.resume)

    run_if_missing([tcn_dir / "tcn_encoder.pt", tcn_dir / "train_history.json"], [
        py, str(SCRIPT_PATHS["train_tcn_encoder"]),
        "--waves", str(tensor_npz["train"]),
        "--out", str(tcn_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--embedding-dim", str(args.embedding_dim),
        "--seed", str(args.seed),
        "--log-target",
    ], args.resume)

    for split_name in ["train", "valid", "test"]:
        run_if_missing([embed_csv[split_name]], [
            py, str(SCRIPT_PATHS["export_tcn_encoder"]),
            "--model", str(tcn_dir),
            "--waves", str(tensor_npz[split_name]),
            "--out", str(embed_csv[split_name]),
        ], args.resume)
        run_if_missing([hybrid_csv[split_name]], [
            py, str(SCRIPT_PATHS["merge_features_and_embeddings"]),
            "--features", str(feature_csv[split_name]),
            "--embeddings", str(embed_csv[split_name]),
            "--out", str(hybrid_csv[split_name]),
        ], args.resume)

    run_if_missing([ag_dir / "predictor.pkl", ag_dir / f"test_predictions_{args.ag_name}.csv"], [
        py, str(SCRIPT_PATHS["train_ag_1stage"]),
        "--train", str(hybrid_csv["train"]),
        "--valid", str(hybrid_csv["valid"]),
        "--test", str(hybrid_csv["test"]),
        "--label", args.label,
        "--model-dir", str(ag_dir),
        "--model-name", args.ag_name,
        "--time-limit", str(args.time_limit),
        "--seed", str(args.seed),
        "--log-target",
    ], args.resume)

    test_pred = ag_dir / f"test_predictions_{args.ag_name}.csv"
    run_if_missing([analysis_dir / "summary.txt"], [
        py, str(SCRIPT_PATHS["analyze_regression_preds"]),
        "--in", str(test_pred),
        "--outdir", str(analysis_dir),
        "--fast-ms", str(args.fast_ms),
    ], args.resume)

    feature_importance = ag_dir / f"feature_importance_{args.ag_name}.csv"
    if feature_importance.exists():
        fi_outdir = ROOT / "analysis" / f"feature_importance_{args.ag_name}"
        run_if_missing([fi_outdir / "feature_summary.txt"], [
            py, str(SCRIPT_PATHS["features"]),
            "--in", str(feature_importance),
            "--outdir", str(fi_outdir),
            "--topn", "30",
        ], args.resume)

    print("\nDone.")
    print(f"TCN model: {tcn_dir}")
    print(f"AutoGluon model: {ag_dir}")
    print(f"Analysis: {analysis_dir}")


if __name__ == "__main__":
    main()
