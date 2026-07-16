"""Wrapper to fine-tune English→Prakrit using the en-indic base model."""
import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune English → Prakrit model")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--model-dir", default=os.path.join("iterative_back_translation", "models", "eng_to_prakrit_iterative"))
    parser.add_argument("--train", default=os.path.join("iterative_back_translation", "data", "iteration1_parallel.tsv"))
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL_PATH", "ai4bharat/indictrans2-en-indic-1B"),
        help="HF model id or local checkpoint path for the base IndicTrans2 en-indic model",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local files (offline-safe)",
    )
    parser.add_argument("--test", default="")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Use finetuning_en_to_prakrit.py (swapped lang tags + column reading)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finetune_script = os.path.join(script_dir, "finetuning_en_to_prakrit.py")

    cmd = [
        sys.executable,
        finetune_script,
        "--train",
        args.train,
        "--base_model",
        args.base_model,
        "--model",
        args.model_dir,
        "--epoch",
        str(args.epoch),
        "--test_size",
        str(args.test_size),
        "--seed",
        str(args.seed),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.test:
        cmd.extend(["--test", args.test])
    run(cmd)


if __name__ == "__main__":
    main()
