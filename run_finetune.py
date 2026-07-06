import argparse
import os
import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--model-dir", default=os.path.join("models", "prakrit_to_eng"))
    parser.add_argument("--train", default=os.path.join("data", "prakrit_eng.clean.tsv"))
    parser.add_argument("--test", default="")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true", help="Use PEFT/LoRA fine-tuning")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    cmd = [
        sys.executable,
        os.path.join("finetuning_and_inferencing_using_indictrans2_models", "finetuning_using_indictrans2_model.py"),
        "--train",
        args.train,
        "--model",
        args.model_dir,
        "--epoch",
        str(args.epoch),
        "--test_size",
        str(args.test_size),
        "--seed",
        str(args.seed),
    ]
    if args.test:
        cmd.extend(["--test", args.test])
    if args.use_lora:
        cmd.append("--use-lora")
    run(cmd)


if __name__ == "__main__":
    main()
