# Prakrit ↔ English Machine Translation

Neural machine translation between **Maharashtri Prakrit** (Devanagari) and **English**, built by fine-tuning [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (1B) with iterative back-translation to augment a small parallel corpus.

## Project Structure

```
prakrit_en_mt/
├── data/                           # Parallel corpus & test sets
│   ├── final_train_clean.tsv       #   ~1,300 Prakrit–English training pairs (TSV)
│   ├── prakrit_eng.clean.tsv       #   Full cleaned parallel corpus (~1,475 pairs)
│   ├── 100_test_pra.txt            #   100 Prakrit test sentences
│   └── 100_test_eng.txt            #   100 English reference translations
│
├── finetuning_and_inferencing_using_indictrans2_models/
│   ├── finetuning_using_indictrans2_model.py        # Core fine-tuning logic (Seq2SeqTrainer)
│   ├── inferencing_using_saved_model.py             # Inference with a fine-tuned checkpoint
│   └── inferencing_using_indictrans2_baseline_models.py  # Baseline (no fine-tuning) inference
│
├── iterative_back_translation/     # Iterative back-translation pipeline
│   ├── data/                       #   Monolingual Prakrit sets (pra_set_1–5, prakrit_50k)
│   │                               #   & corresponding generated English sets (eng_dataset_1–5)
│   ├── scripts/
│   │   ├── run_inference_iterative.py       # Translate mono Prakrit → English & save TSV
│   │   ├── prepare_backtranslation_data.py  # Merge Prakrit + generated English into TSV
│   │   ├── run_finetune_pr_en.py            # Continue fine-tuning Prakrit → English
│   │   ├── run_finetune_en_pr.py            # Continue fine-tuning English → Prakrit
│   │   ├── finetuning_pr_to_eng_iterative.py  # Training script (Prakrit → English)
│   │   ├── finetuning_en_to_prakrit.py        # Training script (English → Prakrit)
│   │   └── slurm_*.sh              # SLURM job scripts for each step
│   └── logs/                       #   Training & inference logs
│
├── run_finetune.py                 # CLI wrapper — fine-tune Prakrit → English
├── run_inference.py                # CLI wrapper — run inference with fine-tuned model
├── run_inference_baseline.py       # CLI wrapper — baseline (pre-trained) inference
├── run_bleu.py                     # Compute SacreBLEU score (hypothesis vs reference)
│
├── scripts/                        # SLURM job scripts (top-level)
│   ├── slurm_finetune.sh
│   ├── slurm_inference.sh
│   ├── slurm_bleu.sh
│   └── slurm_inference_baseline.sh
│
├── inferences/                     # Inference outputs (generated translations)
├── logs/                           # SLURM job logs
├── requirements.txt                # Python dependencies
└── .gitignore
```

## Approach Overview

1. **Baseline** — Evaluate the pre-trained IndicTrans2 (indic → en, 1B) on 100 Prakrit test sentences without any fine-tuning.
2. **Supervised fine-tuning** — Fine-tune IndicTrans2 on the ~1,300-pair parallel corpus (`final_train_clean.tsv`) for Prakrit → English.
3. **Iterative back-translation** — Use the fine-tuned model to translate ~50k monolingual Prakrit sentences into English, pair them into synthetic parallel data, and continue fine-tuning both directions (Prakrit ↔ English) over multiple iterations.
4. **Evaluation** — Compute SacreBLEU on the held-out 100-sentence test set after each round.

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Also install IndicTransToolkit (see requirements.txt comments for git URLs)
```

### 2. Download the base model

```bash
# Cache IndicTrans2 indic→en (1B) locally (needed once, before cluster jobs)
python -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-indic-en-1B', trust_remote_code=True)"
```

### 3. Run baseline inference

```bash
python run_inference_baseline.py --local-files-only
# Output → inferences/prakrit_to_eng.baseline.txt
```

### 4. Fine-tune (Prakrit → English)

```bash
python run_finetune.py --epoch 30 --train data/final_train_clean.tsv --test-size 0
# Or on SLURM:
sbatch scripts/slurm_finetune.sh
```

### 5. Run inference with the fine-tuned model

```bash
python run_inference.py --model-dir models/prakrit_to_eng_v5-final --local-files-only
# Output → inferences/prakrit_to_english_v5.txt
```

### 6. Evaluate (SacreBLEU)

```bash
python run_bleu.py --hyp inferences/prakrit_to_english_v5.txt --ref data/100_test_eng.txt
```

### 7. Iterative back-translation

```bash
# a) Generate synthetic English from monolingual Prakrit
python iterative_back_translation/scripts/run_inference_iterative.py \
    --input iterative_back_translation/data/prakrit_50k.txt \
    --model-dir models/prakrit_to_eng_v5-final \
    --output iterative_back_translation/data/iteration1_parallel.tsv \
    --local-files-only

# b) Continue fine-tuning Prakrit → English on augmented data
python iterative_back_translation/scripts/run_finetune_pr_en.py \
    --train iterative_back_translation/data/iteration1_parallel.tsv \
    --epoch 30 --local-files-only

# c) Continue fine-tuning English → Prakrit
python iterative_back_translation/scripts/run_finetune_en_pr.py \
    --train iterative_back_translation/data/iteration1_parallel.tsv \
    --epoch 30 --local-files-only

# Repeat (a)–(c) with updated models for subsequent iterations.
```

## Hardware

All training and inference jobs are designed for a **SLURM-managed GPU cluster** (128 GB RAM, 48 GPU shards). The SLURM scripts under `scripts/` and `iterative_back_translation/scripts/` can be submitted directly with `sbatch`.

