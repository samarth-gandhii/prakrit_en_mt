import argparse
import os
import subprocess
import sys
import pandas as pd


def run(cmd):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--output", default=os.path.join("inferences", "prakrit_to_eng.baseline.txt"))
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    input_file = os.path.join("inferences", f"input_prakrit_{args.sample_count}.txt")

    os.makedirs(os.path.dirname(os.path.abspath(input_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Use pre-existing local input file (cluster is offline)
    if not os.path.exists(input_file):
        print(f"ERROR: Input file {input_file} not found. Generate it locally before submitting.")
        sys.exit(1)

    cmd = [
        sys.executable,
        os.path.join("finetuning_and_inferencing_using_indictrans2_models", "inferencing_using_indictrans2_baseline_models.py"),
        "--input",
        input_file,
        "--output",
        args.output,
    ]
    run(cmd)

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
