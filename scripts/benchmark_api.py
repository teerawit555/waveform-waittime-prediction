"""Latency benchmarks for NEUROSETTLE prediction APIs.

The ``web-upload`` mode follows the same API workflow as the Prediction page:

1. upload a CSV through ``/api/upload``;
2. start an asynchronous job through ``/api/predict``;
3. poll ``/api/jobs/<job_id>`` until completion.

Its reported pipeline latency is the sum of the six compute stages requested
for comparison. Upload, queueing, plot generation, and result assembly are not
included in that metric.
"""

from __future__ import annotations

import argparse
import builtins
import datetime
import json
import random
import statistics
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


REPORT_WIDTH = 170
API_URL = "http://127.0.0.1:5000/api"
MODEL_NAME = "TCN_aug_weighted_v1"
REQUEST_TIMEOUT_SECONDS = 120.0
WEB_PIPELINE_STAGES = (
    "prepare_data",
    "feature_extraction",
    "tensor_build",
    "tcn_embedding",
    "feature_merge",
    "model_prediction",
)

# Override print so console output can also be written to the report file.
output_lines: list[str] = []


def print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(arg) for arg in args) + end
    builtins.print(*args, **kwargs)
    output_lines.append(msg.rstrip("\r\n"))


def generate_mock_waveform(length: int = 1000) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(length)]


def _execute_json_request(req: urllib.request.Request, endpoint: str):
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            return time.perf_counter() - started_at, json.loads(body)
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started_at
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"\n[Error] Endpoint {endpoint} returned status {exc.code}: {error_body}")
        return elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(f"\n[Error] Connection failed for endpoint {endpoint}: {exc}")
        return elapsed, None


def send_request(endpoint: str, payload, method: str = "POST"):
    url = f"{API_URL}/{endpoint.lstrip('/')}"
    if method == "GET":
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=method,
        )
    return _execute_json_request(request, endpoint)


def resolve_model_name(requested_model_name: str | None) -> str:
    """Use an explicit model or discover the backend's public default model."""
    if requested_model_name:
        return requested_model_name

    _, response = send_request("models", None, method="GET")
    if response and response.get("default_model"):
        return str(response["default_model"])
    raise RuntimeError(
        "Could not discover the backend default model. "
        "Start the backend or pass --model-name explicitly."
    )


def upload_csv(csv_path: Path):
    """Upload a CSV with the multipart shape used by the browser frontend."""
    boundary = f"----NEUROSETTLEBenchmark{uuid.uuid4().hex}"
    safe_name = csv_path.name.replace('"', "_")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8")
    body += csv_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    endpoint = "upload"
    request = urllib.request.Request(
        f"{API_URL}/{endpoint}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _execute_json_request(request, endpoint)


def percentile_nearest_rank(data: list[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    rank = max(1, int((percentile / 100) * len(sorted_data) + 0.999999999))
    return sorted_data[min(rank, len(sorted_data)) - 1]


def print_stats(name: str, latencies: list[float]) -> None:
    if not latencies:
        print(f"{name:<38} | Failed to retrieve data")
        return

    avg = statistics.mean(latencies) * 1000
    median = statistics.median(latencies) * 1000
    p95 = percentile_nearest_rank(latencies, 95) * 1000
    minimum = min(latencies) * 1000
    maximum = max(latencies) * 1000
    std = statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0
    print(
        f"{name:<38} | Runs: {len(latencies):3d} | Mean: {avg:9.2f}ms "
        f"| Median: {median:9.2f}ms | P95: {p95:9.2f}ms "
        f"| Min: {minimum:9.2f}ms | Max: {maximum:9.2f}ms | Std: {std:8.2f}ms"
    )


def save_report(output_path: str) -> None:
    if not output_path:
        return
    out_path = Path(output_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        builtins.print(f"\nSaved benchmark report to: {out_path.resolve()}")
    except Exception as exc:
        builtins.print(f"\n[WARN] Failed to write report to {out_path}: {exc}")


def load_sync_waveforms(csv_path: Path, needed_count: int) -> list[dict]:
    waveforms_pool: list[dict] = []
    if csv_path.exists():
        print(f"Loading real waveforms from {csv_path}...")
        try:
            import pandas as pd

            dataframe = pd.read_csv(csv_path)
            required = {"wave_id", "sample", "value"}
            if not required.issubset(dataframe.columns):
                raise ValueError(f"CSV missing required columns: {required - set(dataframe.columns)}")

            dataframe = dataframe.sort_values(by=["wave_id", "sample"])
            for wave_id, group in dataframe.groupby("wave_id"):
                waveforms_pool.append(
                    {
                        "wave_id": str(wave_id),
                        "data": group["value"].astype(float).tolist(),
                    }
                )
                if len(waveforms_pool) >= needed_count:
                    break
            print(f"Successfully loaded {len(waveforms_pool)} real waveforms from {csv_path}.")
        except Exception as exc:
            print(f"[WARN] Failed to load real waveforms from CSV: {exc}. Falling back to mock data.")
    else:
        print(f"CSV file not found at {csv_path}. Falling back to mock data.")

    if len(waveforms_pool) < needed_count:
        shortage = needed_count - len(waveforms_pool)
        print(f"Generating {shortage} mock waveforms to reach required pool size of {needed_count}...")
        for index in range(shortage):
            waveforms_pool.append(
                {"wave_id": f"mock_{index}", "data": generate_mock_waveform()}
            )
    return waveforms_pool


def run_sync_benchmark(csv_path: Path) -> None:
    needed_count = 731  # 1 warm-up + 30 singles + 200 batch-10 + 500 batch-50
    waveforms_pool = load_sync_waveforms(csv_path, needed_count)
    print("-" * REPORT_WIDTH)

    print("Performing warm-up request to load the model in memory...")
    send_request(
        "predict-sync",
        {
            "waveform": waveforms_pool[0]["data"],
            "dt_ms": 0.01,
            "model_name": MODEL_NAME,
        },
    )
    print("Warm-up complete!\n")

    print("Benchmarking /predict-sync (Single waveform, 30 runs)...")
    sync_latencies: list[float] = []
    for index in range(30):
        wave = waveforms_pool[1 + index]
        elapsed, response = send_request(
            "predict-sync",
            {"waveform": wave["data"], "dt_ms": 0.01, "model_name": MODEL_NAME},
        )
        if response and "pred_wait_time_ms" in response:
            sync_latencies.append(elapsed)
        time.sleep(0.05)

    print("Benchmarking /predict-batch-sync (Batch size 10, 20 runs)...")
    batch10_latencies: list[float] = []
    for index in range(20):
        start_index = 31 + (index * 10)
        waveforms = waveforms_pool[start_index : start_index + 10]
        elapsed, response = send_request(
            "predict-batch-sync",
            {"waveforms": waveforms, "dt_ms": 0.01, "model_name": MODEL_NAME},
        )
        if response and "predictions" in response:
            batch10_latencies.append(elapsed)
        time.sleep(0.05)

    print("Benchmarking /predict-batch-sync (Batch size 50, 10 runs)...")
    batch50_latencies: list[float] = []
    for index in range(10):
        start_index = 231 + (index * 50)
        waveforms = waveforms_pool[start_index : start_index + 50]
        elapsed, response = send_request(
            "predict-batch-sync",
            {"waveforms": waveforms, "dt_ms": 0.01, "model_name": MODEL_NAME},
        )
        if response and "predictions" in response:
            batch50_latencies.append(elapsed)
        time.sleep(0.05)

    print("Benchmarking /health (Service heartbeat, 30 runs)...")
    health_latencies: list[float] = []
    for _ in range(30):
        elapsed, response = send_request("health", None, method="GET")
        if response and response.get("ok"):
            health_latencies.append(elapsed)
        time.sleep(0.02)

    print("\n" + "=" * REPORT_WIDTH)
    print("SYNCHRONOUS API BENCHMARK RESULTS SUMMARY")
    print("=" * REPORT_WIDTH)
    print_stats("Single Waveform (/predict-sync)", sync_latencies)
    print_stats("Batch of 10 (/predict-batch-sync)", batch10_latencies)
    print_stats("Batch of 50 (/predict-batch-sync)", batch50_latencies)
    print_stats("Heartbeat check (/health)", health_latencies)

    if sync_latencies:
        print(
            f"{'Single Waveform (per waveform cost)':<38} | Avg: "
            f"{statistics.mean(sync_latencies) * 1000:9.2f}ms per wave"
        )
    if batch10_latencies:
        print(
            f"{'Batch of 10 (per waveform cost)':<38} | Avg: "
            f"{statistics.mean(batch10_latencies) * 100:9.2f}ms per wave"
        )
    if batch50_latencies:
        print(
            f"{'Batch of 50 (per waveform cost)':<38} | Avg: "
            f"{statistics.mean(batch50_latencies) * 20:9.2f}ms per wave"
        )
    print("=" * REPORT_WIDTH)


def build_web_upload_subsets(
    source_csv: Path,
    output_dir: Path,
    wave_counts: tuple[int, ...] = (1, 10, 50),
) -> dict[int, Path]:
    """Create whole-wave CSV subsets for browser-equivalent upload tests."""
    import pandas as pd

    if not source_csv.is_file():
        raise FileNotFoundError(f"Web upload benchmark CSV not found: {source_csv}")
    if source_csv.suffix.lower() != ".csv":
        raise ValueError("Web upload benchmark currently requires a CSV file.")

    dataframe = pd.read_csv(source_csv, low_memory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_count = max(wave_counts)
    subsets: dict[int, Path] = {}

    long_columns = {"wave_id", "sample", "time_ms", "value"}
    if long_columns.issubset(dataframe.columns):
        wave_ids = dataframe["wave_id"].dropna().drop_duplicates().tolist()
        if len(wave_ids) < maximum_count:
            raise ValueError(
                f"CSV contains {len(wave_ids)} waves, but {maximum_count} are required."
            )
        for count in wave_counts:
            selected_ids = set(wave_ids[:count])
            subset = dataframe[dataframe["wave_id"].isin(selected_ids)].copy()
            subset_path = output_dir / f"web_upload_{count}_waves.csv"
            subset.to_csv(subset_path, index=False)
            subsets[count] = subset_path
        print(f"Prepared long-format web upload subsets from {len(wave_ids)} available waves.")
        return subsets

    value_columns = [column for column in dataframe.columns if str(column).endswith(":")]
    if "Signal" in dataframe.columns and value_columns:
        if len(value_columns) < maximum_count:
            raise ValueError(
                f"CSV contains {len(value_columns)} wide-format waves, but {maximum_count} are required."
            )
        for count in wave_counts:
            subset = dataframe.loc[:, ["Signal", *value_columns[:count]]].copy()
            subset_path = output_dir / f"web_upload_{count}_waves.csv"
            subset.to_csv(subset_path, index=False)
            subsets[count] = subset_path
        print(f"Prepared wide-format web upload subsets from {len(value_columns)} available waves.")
        return subsets

    raise ValueError(
        "Invalid waveform CSV schema. Expected long columns "
        "wave_id/sample/time_ms/value or wide columns Signal plus names ending in ':'."
    )


def run_web_upload_once(
    csv_path: Path,
    expected_wave_count: int,
    poll_interval: float,
    job_timeout: float,
) -> float:
    _, upload_response = upload_csv(csv_path)
    if not upload_response or not upload_response.get("upload_id"):
        raise RuntimeError("CSV upload failed or returned no upload_id.")

    _, predict_response = send_request(
        "predict",
        {
            "upload_id": upload_response["upload_id"],
            "model_name": MODEL_NAME,
        },
    )
    if not predict_response or not predict_response.get("job_id"):
        raise RuntimeError("Prediction request failed or returned no job_id.")

    job_id = predict_response["job_id"]
    deadline = time.monotonic() + job_timeout
    while time.monotonic() < deadline:
        _, job = send_request(f"jobs/{job_id}", None, method="GET")
        if not job:
            raise RuntimeError(f"Could not read prediction job {job_id}.")

        status = job.get("status")
        if status == "failed":
            detail = job.get("error") or job.get("message") or "unknown error"
            raise RuntimeError(f"Prediction job {job_id} failed: {detail}")
        if status == "completed":
            result = job.get("result") or {}
            actual_count = int(result.get("total_waves") or 0)
            if actual_count != expected_wave_count:
                raise RuntimeError(
                    f"Prediction returned {actual_count} waves; expected {expected_wave_count}."
                )

            timings = result.get("stage_timings") or {}
            missing = [stage for stage in WEB_PIPELINE_STAGES if stage not in timings]
            if missing:
                raise RuntimeError(
                    "Prediction result is missing required stage timings: " + ", ".join(missing)
                )
            try:
                stage_values = [float(timings[stage]) for stage in WEB_PIPELINE_STAGES]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Prediction stage timings contain a non-numeric value.") from exc
            if any(value < 0 for value in stage_values):
                raise RuntimeError("Prediction stage timings cannot be negative.")
            return sum(stage_values)

        if status not in {"queued", "running"}:
            raise RuntimeError(f"Prediction job {job_id} returned unexpected status: {status}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Prediction job {job_id} exceeded the {job_timeout:.1f}s timeout.")


def run_web_upload_benchmark(
    source_csv: Path,
    run_counts: dict[int, int],
    poll_interval: float,
    job_timeout: float,
    warm_up: bool,
) -> tuple[dict[int, list[float]], dict[int, int]]:
    if poll_interval <= 0:
        raise ValueError("Poll interval must be greater than zero.")
    if job_timeout <= 0:
        raise ValueError("Job timeout must be greater than zero.")
    if any(value < 0 for value in run_counts.values()):
        raise ValueError("Web benchmark run counts cannot be negative.")

    with tempfile.TemporaryDirectory(prefix="neurosettle_web_benchmark_") as temp_dir:
        subsets = build_web_upload_subsets(source_csv, Path(temp_dir))
        if warm_up:
            print("Performing one web-upload pipeline warm-up (not included in results)...")
            warm_up_seconds = run_web_upload_once(
                subsets[1],
                expected_wave_count=1,
                poll_interval=poll_interval,
                job_timeout=job_timeout,
            )
            print(f"Warm-up complete: six-stage pipeline {warm_up_seconds * 1000:.2f}ms\n")

        results: dict[int, list[float]] = {1: [], 10: [], 50: []}
        failures: dict[int, int] = {1: 0, 10: 0, 50: 0}
        for wave_count in (1, 10, 50):
            requested_runs = run_counts[wave_count]
            consecutive_failures = 0
            print(
                f"Benchmarking web upload pipeline ({wave_count} wave"
                f"{'s' if wave_count != 1 else ''}, {requested_runs} runs)..."
            )
            for run_number in range(1, requested_runs + 1):
                try:
                    pipeline_seconds = run_web_upload_once(
                        subsets[wave_count],
                        expected_wave_count=wave_count,
                        poll_interval=poll_interval,
                        job_timeout=job_timeout,
                    )
                except Exception as exc:
                    failures[wave_count] += 1
                    consecutive_failures += 1
                    print(f"  Run {run_number:>2}/{requested_runs}: FAILED - {exc}")
                    if consecutive_failures >= 3:
                        print("  Aborting this batch after 3 consecutive failed runs.")
                        break
                    continue

                consecutive_failures = 0
                results[wave_count].append(pipeline_seconds)
                print(
                    f"  Run {run_number:>2}/{requested_runs}: "
                    f"six-stage pipeline {pipeline_seconds * 1000:.2f}ms"
                )
        return results, failures


def print_web_upload_summary(
    results: dict[int, list[float]],
    failures: dict[int, int],
) -> None:
    print("\n" + "=" * REPORT_WIDTH)
    print("WEB UPLOAD PIPELINE BENCHMARK RESULTS SUMMARY")
    print("=" * REPORT_WIDTH)
    print("Included: Prepare Data + Feature Extraction + Tensor Build + TCN Embedding + Feature Merge + Model Prediction")
    print("Excluded: CSV upload + queue/waiting + Plot Generation + Result Assembly")
    print("-" * REPORT_WIDTH)
    print_stats("Single Waveform (web upload)", results[1])
    print_stats("Batch of 10 (web upload)", results[10])
    print_stats("Batch of 50 (web upload)", results[50])
    if any(failures.values()):
        print(
            "Failed runs                             | "
            f"1 wave: {failures[1]} | 10 waves: {failures[10]} | 50 waves: {failures[50]}"
        )

    labels = {
        1: "Single Waveform (per waveform cost)",
        10: "Batch of 10 (per waveform cost)",
        50: "Batch of 50 (per waveform cost)",
    }
    for wave_count in (1, 10, 50):
        values = results[wave_count]
        if values:
            average_per_wave_ms = statistics.mean(values) * 1000 / wave_count
            print(
                f"{labels[wave_count]:<38} | Avg: "
                f"{average_per_wave_ms:9.2f}ms per wave"
            )
    print("=" * REPORT_WIDTH)


def main() -> None:
    global API_URL, MODEL_NAME, REQUEST_TIMEOUT_SECONDS

    default_out_dir = Path("benchmark_reports")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out_path = default_out_dir / f"benchmark_results_{timestamp}.txt"

    parser = argparse.ArgumentParser(description="Latency benchmark tool for prediction APIs.")
    parser.add_argument(
        "--csv",
        default="data/generated/sample_100_waves.csv",
        help="Path to a real waveform CSV containing at least 50 waves",
    )
    parser.add_argument("--out", default=str(default_out_path), help="Path to save results TXT file")
    parser.add_argument(
        "--mode",
        choices=("sync", "web-upload", "all"),
        default="web-upload",
        help="Benchmark synchronous APIs, the browser-style upload pipeline, or both",
    )
    parser.add_argument("--api-url", default=API_URL, help="Backend API root, including /api")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Internal model name; defaults to default_model returned by /api/models",
    )
    parser.add_argument("--web-runs-1", type=int, default=30, help="Measured web-upload runs for 1 wave")
    parser.add_argument("--web-runs-10", type=int, default=20, help="Measured web-upload runs for 10 waves")
    parser.add_argument("--web-runs-50", type=int, default=10, help="Measured web-upload runs for 50 waves")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between job status polls")
    parser.add_argument("--job-timeout", type=float, default=900.0, help="Maximum seconds to wait for each job")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Socket timeout in seconds for each HTTP request",
    )
    parser.add_argument(
        "--skip-web-warmup",
        action="store_true",
        help="Skip the unmeasured 1-wave warm-up job",
    )
    args = parser.parse_args()

    API_URL = args.api_url.rstrip("/")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than zero")
    REQUEST_TIMEOUT_SECONDS = args.request_timeout

    print("=" * REPORT_WIDTH)
    print("NEUROSETTLE API LATENCY BENCHMARK TOOL")
    print("=" * REPORT_WIDTH)
    print(f"Benchmark Mode: {args.mode}")
    print(f"Targeting Backend API: {API_URL}")
    print(f"Source CSV: {Path(args.csv)}")
    print("-" * REPORT_WIDTH)

    exit_code = 0
    try:
        MODEL_NAME = resolve_model_name(args.model_name)
        print(f"Targeting Model Name: {MODEL_NAME}")
        print("-" * REPORT_WIDTH)
        if args.mode in {"sync", "all"}:
            run_sync_benchmark(Path(args.csv))
        if args.mode in {"web-upload", "all"}:
            results, failures = run_web_upload_benchmark(
                source_csv=Path(args.csv),
                run_counts={
                    1: args.web_runs_1,
                    10: args.web_runs_10,
                    50: args.web_runs_50,
                },
                poll_interval=args.poll_interval,
                job_timeout=args.job_timeout,
                warm_up=not args.skip_web_warmup,
            )
            print_web_upload_summary(results, failures)
            if any(failures.values()):
                exit_code = 1
    except Exception as exc:
        print(f"\n[ERROR] Benchmark failed: {exc}")
        exit_code = 1
    finally:
        save_report(args.out)

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
