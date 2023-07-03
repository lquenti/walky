import sys
import subprocess
import json

command = sys.argv[1]
output_file = sys.argv[2]

warmup_runs = 3
measurement_runs = 10
solution_weight = []

# Warmup runs
for _ in range(warmup_runs):
    subprocess.run(command, shell=True, capture_output=True)

# Measurement runs
for _ in range(measurement_runs):
    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(output.stderr)
    lines = output.stdout.split("\n")
    for line in lines:
        if line.startswith("Christofides solution weight:"):
            weight = float(line.split(":")[1].strip())
            solution_weight.append(weight)
            break

results = {
    "solution_weight": solution_weight,
    "mean": sum(solution_weight) / len(solution_weight),
    "median": sorted(solution_weight)[len(solution_weight) // 2]
}

with open(output_file, "w") as f:
    json.dump(results, f)
