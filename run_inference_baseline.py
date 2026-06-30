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

# 🔴 DECOUPLING FIX: If the input test file doesn't exist, create it natively right here!
    if not os.path.exists(input_file):
        print(f"Input file {input_file} missing. Generating it dynamically from Hugging Face dataset...")
        hf_path = "hf://datasets/VIITPune/Deshika-Maharashtri_Prakrit_to_English_Parallel_Corpus/prakrit_translation.csv"
        df = pd.read_csv(hf_path)
        prakrit_lines = df["prakrit"].head(args.sample_count).fillna("").astype(str).tolist()
        
        with open(input_file, "w", encoding="utf-8") as f_in:
            f_write_lines = [line.replace("\n", " ").strip() for line in prakrit_lines]
            f_in.write("\n".join(f_write_lines) + "\n")

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
