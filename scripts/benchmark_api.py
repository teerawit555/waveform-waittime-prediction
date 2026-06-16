# ==============================================================================
# This file benchmarks the performance of the synchronous prediction APIs.
# It supports reading real waveform data from a CSV file (e.g. data_for_pred.csv)
# and automatically logs the console output to benchmark_results.txt.
# ==============================================================================
import time
import urllib.request
import urllib.error
import json
import statistics
import random
import builtins
import argparse
import datetime
from pathlib import Path

# Override print to capture output for saving to file
output_lines = []

def print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(str(arg) for arg in args) + end
    builtins.print(*args, **kwargs)
    output_lines.append(msg.rstrip('\r\n'))

# API Server Configuration
API_URL = "http://127.0.0.1:5000/api"
MODEL_NAME = "Aug_best_old_data_v1"  # Use existing local model name

# Generate a mock waveform with 1000 points (normalized values between -1.0 and 1.0)
def generate_mock_waveform(length=1000):
    return [random.uniform(-1.0, 1.0) for _ in range(length)]

def send_request(endpoint, payload, method="POST"):
    url = f"{API_URL}/{endpoint}"
    if method == "GET":
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read().decode("utf-8")
            elapsed = time.perf_counter() - t0
            return elapsed, json.loads(body)
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        err_msg = e.read().decode("utf-8")
        print(f"\n[Error] Endpoint {endpoint} returned status {e.code}: {err_msg}")
        return elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n[Error] Connection failed to {url}: {e}")
        return elapsed, None

def print_stats(name, latencies):
    if not latencies:
        print(f"{name:<30} | Failed to retrieve data")
        return
    
    avg = statistics.mean(latencies) * 1000
    med = statistics.median(latencies) * 1000
    p95 = percentiles(latencies, 95) * 1000
    min_val = min(latencies) * 1000
    max_val = max(latencies) * 1000
    std = statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0
    
    print(f"{name:<35} | Mean: {avg:6.2f}ms | Median: {med:6.2f}ms | P95: {p95:6.2f}ms | Min: {min_val:6.2f}ms | Max: {max_val:6.2f}ms | Std: {std:5.2f}ms")

def percentiles(data, percentile):
    size = len(data)
    if size == 0:
        return 0
    sorted_data = sorted(data)
    return sorted_data[int(math_ceil((percentile / 100) * size)) - 1]

def math_ceil(x):
    return int(x) + (1 if x % 1 > 0 else 0)

def main():
    default_out_dir = Path("benchmark_reports")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out_path = default_out_dir / f"benchmark_results_{timestamp}.txt"

    parser = argparse.ArgumentParser(description="Latency benchmark tool for prediction APIs.")
    parser.add_argument("--csv", default="data/data_for_pred.csv", help="Path to real data CSV")
    parser.add_argument("--out", default=str(default_out_path), help="Path to save results txt file")
    args = parser.parse_args()

    print("=" * 110)
    print("NEUROSETTLE API LATENCY BENCHMARK TOOL")
    print("=" * 110)
    print(f"Targeting Backend API: {API_URL}")
    print(f"Targeting Model Name: {MODEL_NAME}")
    print("-" * 110)

    # Load real waveforms from CSV if available, otherwise fallback to mock
    needed_count = 731 # 1 + 30 + 200 + 500 = 731
    waveforms_pool = []
    
    if args.csv:
        csv_path = Path(args.csv)
        if csv_path.exists():
            print(f"Loading real waveforms from {csv_path}...")
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                required = {"wave_id", "sample", "value"}
                if not required.issubset(df.columns):
                    raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")
                
                df = df.sort_values(by=["wave_id", "sample"])
                # Extract waves grouped by wave_id
                for wid, group in df.groupby("wave_id"):
                    waveforms_pool.append({
                        "wave_id": str(wid),
                        "data": group["value"].astype(float).tolist()
                    })
                    if len(waveforms_pool) >= needed_count:
                        break
                print(f"Successfully loaded {len(waveforms_pool)} real waveforms from {csv_path}.")
            except Exception as e:
                print(f"[WARN] Failed to load real waveforms from CSV: {e}. Falling back to mock data.")
        else:
            print(f"CSV file not found at {csv_path}. Falling back to mock data.")

    # Pad with mock data if we don't have enough waves
    if len(waveforms_pool) < needed_count:
        shortage = needed_count - len(waveforms_pool)
        print(f"Generating {shortage} mock waveforms to reach required pool size of {needed_count}...")
        for i in range(shortage):
            waveforms_pool.append({
                "wave_id": f"mock_{i}",
                "data": generate_mock_waveform()
            })

    print("-" * 110)

    # 1. Warm-up request
    print("Performing warm-up request to load the model in memory...")
    dummy_payload = {
        "waveform": waveforms_pool[0]["data"],
        "dt_ms": 0.01,
        "model_name": MODEL_NAME
    }
    send_request("predict-sync", dummy_payload)
    print("Warm-up complete!\n")

    # 2. Benchmark /predict-sync (Single waveform)
    print("Benchmarking /predict-sync (Single waveform, 30 runs)...")
    sync_latencies = []
    for i in range(30):
        wave = waveforms_pool[1 + i]
        payload = {
            "waveform": wave["data"],
            "dt_ms": 0.01,
            "model_name": MODEL_NAME
        }
        elapsed, resp = send_request("predict-sync", payload)
        if resp and "pred_wait_time_ms" in resp:
            sync_latencies.append(elapsed)
        time.sleep(0.05)  # Brief pause between calls

    # 3. Benchmark /predict-batch-sync (Batch of 10)
    print("Benchmarking /predict-batch-sync (Batch size 10, 20 runs)...")
    batch10_latencies = []
    for i in range(20):
        # Slice waves from pool
        start_idx = 31 + (i * 10)
        end_idx = start_idx + 10
        waveforms = waveforms_pool[start_idx:end_idx]
        
        payload = {
            "waveforms": waveforms,
            "dt_ms": 0.01,
            "model_name": MODEL_NAME
        }
        elapsed, resp = send_request("predict-batch-sync", payload)
        if resp and "predictions" in resp:
            batch10_latencies.append(elapsed)
        time.sleep(0.05)

    # 4. Benchmark /predict-batch-sync (Batch of 50)
    print("Benchmarking /predict-batch-sync (Batch size 50, 10 runs)...")
    batch50_latencies = []
    for i in range(10):
        # Slice waves from pool
        start_idx = 231 + (i * 50)
        end_idx = start_idx + 50
        waveforms = waveforms_pool[start_idx:end_idx]
        
        payload = {
            "waveforms": waveforms,
            "dt_ms": 0.01,
            "model_name": MODEL_NAME
        }
        elapsed, resp = send_request("predict-batch-sync", payload)
        if resp and "predictions" in resp:
            batch50_latencies.append(elapsed)
        time.sleep(0.05)

    # 5. Benchmark /predict (Asynchronous Job initiation latency)
    print("Benchmarking /health (Service heartbeat, 30 runs)...")
    health_latencies = []
    for i in range(30):
        elapsed, resp = send_request("health", None, method="GET")
        if resp and resp.get("ok"):
            health_latencies.append(elapsed)
        time.sleep(0.02)

    # Print Results Summary
    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 110)
    print_stats("Single Waveform (/predict-sync)", sync_latencies)
    print_stats("Batch of 10 (/predict-batch-sync)", batch10_latencies)
    print_stats("Batch of 50 (/predict-batch-sync)", batch50_latencies)
    print_stats("Heartbeat check (/health)", health_latencies)
    
    # Calculate per-waveform cost
    if batch10_latencies:
        avg_10 = (statistics.mean(batch10_latencies) * 1000) / 10
        print(f"{'Batch of 10 (per waveform cost)':<35} | Avg:  {avg_10:6.2f}ms per wave")
    if batch50_latencies:
        avg_50 = (statistics.mean(batch50_latencies) * 1000) / 50
        print(f"{'Batch of 50 (per waveform cost)':<35} | Avg:  {avg_50:6.2f}ms per wave")
    
    print("=" * 110)

    # Save to file if output path is set
    if args.out:
        out_path = Path(args.out)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines) + "\n")
            builtins.print(f"\nSaved benchmark report to: {out_path.resolve()}")
        except Exception as e:
            builtins.print(f"\n[WARN] Failed to write report to {out_path}: {e}")

if __name__ == "__main__":
    main()
