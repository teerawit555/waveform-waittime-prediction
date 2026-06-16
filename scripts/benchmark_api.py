import time
import urllib.request
import urllib.error
import json
import statistics
import random

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
    print("=" * 110)
    print("NEUROSETTLE API LATENCY BENCHMARK TOOL")
    print("=" * 110)
    print(f"Targeting Backend API: {API_URL}")
    print(f"Targeting Model Name: {MODEL_NAME}")
    print("-" * 110)

    # 1. Warm-up request
    print("Performing warm-up request to load the model in memory...")
    dummy_payload = {"waveform": generate_mock_waveform(), "dt_ms": 0.01, "model_name": MODEL_NAME}
    send_request("predict-sync", dummy_payload)
    print("Warm-up complete!\n")

    # 2. Benchmark /predict-sync (Single waveform)
    print("Benchmarking /predict-sync (Single waveform, 30 runs)...")
    sync_latencies = []
    for i in range(30):
        payload = {
            "waveform": generate_mock_waveform(),
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
        waveforms = [
            {"wave_id": f"wave_{j}", "data": generate_mock_waveform()}
            for j in range(10)
        ]
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
        waveforms = [
            {"wave_id": f"wave_{j}", "data": generate_mock_waveform()}
            for j in range(50)
        ]
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
    # Note: We won't trigger the actual pipeline to avoid generating large artifacts,
    # but we'll measure the time to get the queued response / health check.
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

if __name__ == "__main__":
    main()
