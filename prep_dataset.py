"""Pre-tokenize a dataset into the block format train_gpt.py expects.

Mirrors train_gpt.py's tokenize_and_group exactly (concat -> block chunks,
labels = input_ids), so training can point tokenized_dataset_path at the
output and skip preprocessing entirely. Intended to build a shared cache
usable across projects, e.g.:

    python prep_dataset.py \
        --dataset Skylion007/openwebtext \
        --out /workspace/data/tokenized/openwebtext_gpt2_1024
"""

import argparse
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Skylion007/openwebtext")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--tokenizer", default="openai-community/gpt2")
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--validation_split_percentage", type=int, default=5)
    ap.add_argument("--num_proc", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    raw = load_dataset(
        args.dataset,
        args.dataset_config,
        split={
            "train": f"train[{args.validation_split_percentage}%:]",
            "validation": f"train[:{args.validation_split_percentage}%]",
        },
        num_proc=args.num_proc,
    )
    column_names = raw["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]
    block_size = args.block_size

    def tokenize_and_group(examples):
        tokenized = tokenizer(examples[text_column])
        concatenated = {k: list(chain(*tokenized[k])) for k in tokenized.keys()}
        total_length = len(concatenated[list(tokenized.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm = raw.map(
        tokenize_and_group,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc=f"Tokenize + group into {block_size}",
    )
    print({split: len(ds) for split, ds in lm.items()})
    lm.save_to_disk(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
