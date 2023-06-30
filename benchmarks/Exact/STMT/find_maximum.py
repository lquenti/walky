import json
import subprocess
import os
import sys

i = 3
algorithm = sys.argv[1]
parallelism = sys.argv[2]

# cluster conf
PROGRAM = "/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky/target/release/walky"
XML_PATH = "/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky/utils/gen_matrix_fast/results"
MAX_TIME = 5*60

# local testing
#PROGRAM = "/home/lquenti/code/walky/target/release/walky"
#XML_PATH = "/home/lquenti/code/walky/utils/gen_matrix_fast/results/"
#MAX_TIME = .2

OUTPUT_FILE = f"minimum_{algorithm}_{parallelism}.txt"

# since pruning is so random it has to fail twice for us to stop
FAILED_ONCE = False


# Clear out the file before starting the loop
with open(OUTPUT_FILE, 'w') as file:
    pass

while True:
    subprocess.run([
        "hyperfine",
        "--shell=none",
        "--warmup",
        "1",
        "--runs",
        "2",
        "--export-json",
        "results.json",
        #"--show-output",
        f"{PROGRAM} exact -p {parallelism} {algorithm} {XML_PATH}/{i}.xml"
    ])

    with open('results.json', 'r') as file:
        data = json.load(file)
        median = data['results'][0]['median']

    output = f"Median: {median}, i: {i}, algorithm: {algorithm}, parallelism: {parallelism}"
    print(output)
    with open(OUTPUT_FILE, 'a') as file:
        file.write(output + '\n')

    # check if we can stop
    if median > MAX_TIME:
        if FAILED_ONCE:
            sys.exit(0)
        else:
            FAILED_ONCE = True
    else:
        FAILED_ONCE= False

    # cleanup
    i += 1
    os.remove('results.json')

