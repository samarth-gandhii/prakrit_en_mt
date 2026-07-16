"""Run iterative back translation inference (Prakrit→English) and save parallel corpus."""
import argparse
import csv
import os
import subprocess
import sys


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Iterative back translation: infer English from Prakrit, save as parallel corpus TSV"
    )
    parser.add_argument("--input", required=True, help="Path to input Prakrit text file")
    parser.add_argument("--output", default="iteration1_parallel.tsv",
                        help="Path to output parallel TSV (prakrit \\t english)")
    parser.add_argument("--model-dir", required=True,
                        help="Path to fine-tuned Prakrit→English model directory")
    parser.add_argument("--base-model",
                        default=os.environ.get("BASE_MODEL_PATH", "ai4bharat/indictrans2-indic-en-1B"),
                        help="HF model id or local path for base model (used for tokenizer).")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load only from local files (offline mode).")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    output_tsv = os.path.abspath(args.output)
    model_dir = os.path.abspath(args.model_dir)

    # Raw English output goes alongside the TSV
    english_raw = output_tsv.replace(".tsv", "_english_raw.txt")

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # ---------- Step 1: Run inference ----------
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    inference_script = os.path.join(
        repo_root,
        "finetuning_and_inferencing_using_indictrans2_models",
        "inferencing_using_saved_model.py",
    )

    cmd = [
        sys.executable,
        inference_script,
        "--input", input_file,
        "--model", model_dir,
        "--output", english_raw,
        "--base-model", args.base_model,
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    run(cmd)

    # ---------- Step 2: Create parallel corpus TSV ----------
    print("\nCreating parallel corpus TSV...")
    with open(input_file, "r", encoding="utf-8") as f:
        prakrit_lines = [line.strip() for line in f if line.strip()]
    with open(english_raw, "r", encoding="utf-8") as f:
        english_lines = [line.strip() for line in f if line.strip()]

    if len(prakrit_lines) != len(english_lines):
        print(f"WARNING: Line count mismatch! Prakrit: {len(prakrit_lines)}, English: {len(english_lines)}")
        min_len = min(len(prakrit_lines), len(english_lines))
        prakrit_lines = prakrit_lines[:min_len]
        english_lines = english_lines[:min_len]

    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["prakrit", "english"])
        for pr, en in zip(prakrit_lines, english_lines):
            writer.writerow([pr, en])

    print(f"Parallel corpus saved: {output_tsv} ({len(prakrit_lines)} pairs)")
    print(f"Raw English output kept at: {english_raw}")


if __name__ == "__main__":
    main()
