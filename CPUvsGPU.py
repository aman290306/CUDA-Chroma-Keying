import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

with open("cpu_time.txt", "r") as f:
    cpu_times = [float(line.strip()) for line in f.readlines()]

with open("gpu_time.txt", "r") as f:
    gpu_times = [float(line.strip()) for line in f.readlines()]

n = min(len(cpu_times), len(gpu_times))
frames = np.array(range(1, n + 1))
cpu = np.array(cpu_times[:n])
gpu = np.array(gpu_times[:n])

frames_smooth = np.linspace(frames.min(), frames.max(), 300)
cpu_smooth = make_interp_spline(frames, cpu, k=3)(frames_smooth)
gpu_smooth = make_interp_spline(frames, gpu, k=3)(frames_smooth)

plt.figure(figsize=(10, 6))
plt.plot(frames_smooth, cpu_smooth, label="CPU", linewidth=2)
plt.plot(frames_smooth, gpu_smooth, label="GPU", linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("Time per Frame (ms)")
plt.title("CPU vs GPU per-frame processing time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()