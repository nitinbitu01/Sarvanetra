# benchmark_gpu.py
import onnxruntime as ort
import numpy as np
import time

print("="*55)
print("SENTINEL GPU BENCHMARK — RTX 4070")
print("="*55)

ONNX_MODEL  = "yolov8n.onnx"
RUNS        = 100
INPUT_SHAPE = (1, 3, 640, 640)

def benchmark(providers, label):
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            ONNX_MODEL,
            sess_options=sess_options,
            providers=providers
        )
        actual = session.get_providers()[0]
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

        # Warmup — essential for accurate GPU timing
        print(f"\n{label}: warming up (20 runs)...")
        for _ in range(20):
            session.run(None, {input_name: dummy})

        # Timed benchmark
        print(f"{label}: benchmarking ({RUNS} runs)...")
        start = time.perf_counter()
        for _ in range(RUNS):
            session.run(None, {input_name: dummy})
        elapsed = time.perf_counter() - start

        fps = RUNS / elapsed
        ms  = elapsed / RUNS * 1000

        print(f"  Provider : {actual}")
        print(f"  Speed    : {fps:.1f} FPS")
        print(f"  Latency  : {ms:.2f} ms/frame")
        return fps, ms

    except Exception as e:
        print(f"{label}: FAILED — {e}")
        return 0, 0


# CPU baseline
cpu_fps, cpu_ms = benchmark(
    ['CPUExecutionProvider'],
    "CPU baseline"
)

# GPU with CUDA
gpu_fps, gpu_ms = benchmark(
    ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    "GPU RTX 4070 (CUDA)"
)

# TensorRT (may or may not work on CUDA 13.x)
trt_fps, trt_ms = benchmark(
    ['TensorrtExecutionProvider',
     'CUDAExecutionProvider',
     'CPUExecutionProvider'],
    "TensorRT EP"
)

# Summary
print("\n" + "="*55)
print("BENCHMARK RESULTS")
print("="*55)
print(f"{'Backend':<25} {'FPS':>8} {'ms/frame':>10} {'Speedup':>10}")
print("-"*55)
print(f"{'CPU':<25} {cpu_fps:>8.1f} {cpu_ms:>10.2f} {'1.0x':>10}")

if gpu_fps > 0:
    print(f"{'GPU CUDA':<25} {gpu_fps:>8.1f} "
          f"{gpu_ms:>10.2f} "
          f"{gpu_fps/max(cpu_fps,1):>9.1f}x")

if trt_fps > 0:
    print(f"{'TensorRT EP':<25} {trt_fps:>8.1f} "
          f"{trt_ms:>10.2f} "
          f"{trt_fps/max(cpu_fps,1):>9.1f}x")

print("="*55)

# Camera capacity
best_fps = max(gpu_fps, trt_fps, cpu_fps)
best_label = (
    "TensorRT EP" if trt_fps >= gpu_fps
    else "GPU CUDA" if gpu_fps > cpu_fps
    else "CPU"
)

print(f"\nBest backend: {best_label} at {best_fps:.1f} FPS")
print()
print("Camera capacity at 2 FPS processing target:")
print(f"  CPU  : ~{int(cpu_fps/2)} cameras")
if gpu_fps > 0:
    print(f"  GPU  : ~{int(gpu_fps/2)} cameras")
if trt_fps > 0:
    print(f"  TRT  : ~{int(trt_fps/2)} cameras")

print()
print("USE THESE MEASURED NUMBERS IN TENDER DOCS")
print("NOT generic marketing claims")
print("="*55)