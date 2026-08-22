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
    parser.add_argument("--direction", choices=["pra2eng", "eng2pra"], default="pra2eng",
                        help="Translation direction: pra2eng (default) or eng2pra")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for model inference (default: 64)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load only from local files (offline mode).")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    output_tsv = os.path.abspath(args.output)
    model_dir = os.path.abspath(args.model_dir)

    raw_output = output_tsv.replace(".tsv", "_raw_output.txt")

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

    src_lang = "hin_Deva" if args.direction == "pra2eng" else "eng_Latn"
    tgt_lang = "eng_Latn" if args.direction == "pra2eng" else "hin_Deva"

    cmd = [
        sys.executable,
        inference_script,
        "--input", input_file,
        "--model", model_dir,
        "--output", raw_output,
        "--base-model", args.base_model,
        "--src-lang", src_lang,
        "--tgt-lang", tgt_lang,
        "--batch-size", str(args.batch_size),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    run(cmd)

    # ---------- Step 2: Create parallel corpus TSV (Col 0 = Prakrit, Col 1 = English) ----------
    print("\nCreating parallel corpus TSV...")
    with open(input_file, "r", encoding="utf-8") as f:
        input_lines = [line.strip() for line in f if line.strip()]
    with open(raw_output, "r", encoding="utf-8") as f:
        gen_lines = [line.strip() for line in f if line.strip()]

    if len(input_lines) != len(gen_lines):
        print(f"WARNING: Line count mismatch! Input: {len(input_lines)}, Output: {len(gen_lines)}")
        min_len = min(len(input_lines), len(gen_lines))
        input_lines = input_lines[:min_len]
        gen_lines = gen_lines[:min_len]

    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write("prakrit\tenglish\n")
        for inp, gen in zip(input_lines, gen_lines):
            inp_clean = inp.strip().strip('"').strip("'").strip("“").strip("”").strip()
            gen_clean = gen.strip().strip('"').strip("'").strip("“").strip("”").strip()
            if args.direction == "pra2eng":
                pr, en = inp_clean, gen_clean
            else:
                pr, en = gen_clean, inp_clean
            f.write(f"{pr}\t{en}\n")

    print(f"Parallel corpus saved: {output_tsv} ({len(input_lines)} pairs)")
    print(f"Raw model output kept at: {raw_output}")


if __name__ == "__main__":
    main()
