import argparse
import os
import csv
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge Prakrit input and English inference output into a training TSV")
    parser.add_argument("--prakrit", default=os.path.join("..", "data", "prakrit_50k.txt"), help="Path to Prakrit file")
    parser.add_argument("--english", default=os.path.join("..", "data", "english_50k.txt"), help="Path to English file")
    parser.add_argument("--output", default=os.path.join("..", "data", "english_prakrit_50k.tsv"), help="Path to output TSV")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    prakrit_file = os.path.abspath(args.prakrit)
    english_file = os.path.abspath(args.english)
    output_file = os.path.abspath(args.output)

    if not os.path.exists(prakrit_file):
        print(f"ERROR: Prakrit file not found: {prakrit_file}")
        sys.exit(1)

    if not os.path.exists(english_file):
        print(f"ERROR: English file not found: {english_file}")
        sys.exit(1)

    with open(prakrit_file, 'r', encoding='utf-8') as f_pr, open(english_file, 'r', encoding='utf-8') as f_en:
        prakrit_lines = [line.strip() for line in f_pr]
        english_lines = [line.strip() for line in f_en]

    if len(prakrit_lines) != len(english_lines):
        print(f"ERROR: Line count mismatch! Prakrit: {len(prakrit_lines)}, English: {len(english_lines)}")
        sys.exit(1)

    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        # Even though we are training English to Prakrit, we can keep the columns consistent with other scripts
        # or change them depending on how the finetune script reads them.
        # It's usually safe to output 'english' and 'prakrit' columns.
        writer.writerow(['english', 'prakrit'])
        
        for en, pr in zip(english_lines, prakrit_lines):
            # Clean up quotes if present from generation/extraction
            en = en.strip('"')
            pr = pr.strip('"')
            writer.writerow([en, pr])

    print(f"Successfully wrote {len(prakrit_lines)} pairs to {output_file}")


if __name__ == "__main__":
    main()
