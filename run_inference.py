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
    parser.add_argument("--model-dir", default=os.path.join("models", "prakrit_to_eng-final"))
    parser.add_argument("--output", default=os.path.join("inferences", "prakrit_to_eng.txt"))
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    input_file = os.path.join("inferences", f"input_prakrit_{args.sample_count}.txt")
    ref_file = os.path.join("inferences", f"reference_english_{args.sample_count}.txt")

    os.makedirs(os.path.dirname(os.path.abspath(input_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ref_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # hf_path = "hf://datasets/VIITPune/Deshika-Maharashtri_Prakrit_to_English_Parallel_Corpus/prakrit_translation.csv"
    # df = pd.read_csv(hf_path)

    # df["prakrit"].head(args.sample_count).to_csv(input_file, index=False, header=False)
    # df["english"].head(args.sample_count).to_csv(ref_file, index=False, header=False)

# 🔴 DECOUPLING: Always pull data and create the clean files first to break the circular dependency
    print("Fetching and creating clean verification tokens...")
    hf_path = "hf://datasets/VIITPune/Deshika-Maharashtri_Prakrit_to_English_Parallel_Corpus/prakrit_translation.csv"
    df = pd.read_csv(hf_path)

    # 🔴 FIX: Write files natively line-by-line to prevent pandas comma/quoting pollution (")
    prakrit_lines = df["prakrit"].head(args.sample_count).fillna("").astype(str).tolist()
    english_lines = df["english"].head(args.sample_count).fillna("").astype(str).tolist()

    with open(input_file, "w", encoding="utf-8") as f_in:
        f_write_lines = [line.replace("\n", " ").strip() for line in prakrit_lines]
        f_in.write("\n".join(f_write_lines) + "\n")

    with open(ref_file, "w", encoding="utf-8") as f_ref:
        f_write_lines = [line.replace("\n", " ").strip() for line in english_lines]
        f_ref.write("\n".join(f_write_lines) + "\n")

    print(f"Clean input generated: {input_file}")
    print(f"Clean reference generated: {ref_file}")

    # Now verify model path exists before executing inference
    if not os.path.exists(args.model_dir):
        print(f"\n[!] Data setup done, but skipping inference because fine-tuned weights folder '{args.model_dir}' does not exist yet. Run training first.")
        return
        
    cmd = [
        sys.executable,
        os.path.join("finetuning_and_inferencing_using_indictrans2_models", "inferencing_using_saved_model.py"),
        "--input",
        input_file,
        "--model",
        args.model_dir,
        "--output",
        args.output,
    ]
    run(cmd)

    print(f"Input: {input_file}")
    print(f"Reference: {ref_file}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
