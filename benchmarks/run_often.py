import sys
import subprocess
import json

command = sys.argv[1]
output_file = sys.argv[2]

warmup_runs = 3
measurement_runs = 10
elapsed_seconds = []

# Warmup runs
for _ in range(warmup_runs):
    subprocess.run(command, shell=True, capture_output=True)

# Measurement runs
for _ in range(measurement_runs):
    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    lines = output.stdout.split("\n")
    for line in lines:
        if line.startswith("elapsed seconds:"):
            seconds = float(line.split(":")[1].strip())
            elapsed_seconds.append(seconds)
            break

results = {
    "elapsed_seconds": elapsed_seconds,
    "mean": sum(elapsed_seconds) / len(elapsed_seconds),
    "median": sorted(elapsed_seconds)[len(elapsed_seconds) // 2]
}

with open(output_file, "w") as f:
    json.dump(results, f)

