import argparse
import os.path

import torch

from taegan import TAEGAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", "-o", required=True, type=str)
    subparsers = parser.add_subparsers(dest="op")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--src-data-path", "-s", type=str, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--batch-size", "-b", type=int, default=500)
    train_parser.add_argument("--epochs", "-e", type=int, default=30)
    train_parser.add_argument("--warmup-epochs", "-w", type=int, default=9)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--n-rows", "-n", type=int, default=30)
    sample_parser.add_argument("--batch-size", "-b", type=int, default=500)

    recover_parser = subparsers.add_parser("recover")

    return parser.parse_args()


def prepare(args: argparse.Namespace):
    """
    Reads data from args.src_data_path, and produces the following inside args.out_dir:

    1. Prepare train, test, validation sets in `train.csv`, `test.csv`, `val.csv` respectively.
    2. Fit a data transformer that does one-hot encoding for categorical, VGM-decomposition for continuous (with log
       transformation wherever needed), etc., on the train set.
    3. Transform the train set using the fitted transformer into a tensor, saved to `train-data.pt`.
    4. Extract the span information from the fitted transformer in the format of a list of tuples of integer and string
       being either "onehot" or "normal". The integer means the width of the component, and "onehot" means the component
       being one-hot encoded, "normal" means the component being a single continuous component (which would always have
       width 1). The sum of widths should be the same as the number of columns in the transformed tensor.
       Save this list to `span-info.json`.
    5. Save rest of information of the transformer based on the actual transformer implementation if needed, so that
       after generation, data can be inversely transformed.
    """
    ...


def train(args: argparse.Namespace):
    model = TAEGAN(args.out_dir)
    model.train(
        batch_size=args.batch_size, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
    )


def sample(args: argparse.Namespace):
    model = TAEGAN(args.out_dir)
    out = model.generate(batch_size=args.batch_size, n=args.n_rows)
    torch.save(out, os.path.join(args.out_dir, "sampled.pt"))


def recover(args: argparse.Namespace):
    """
    Loads the data transformer information from args.out_dir, and the sampled tensor from args.out_dir/`sampled.pt`,
    inversely transform using the transformer. Save the output to args.out_dir/`sampled-data.csv`.
    """
    ...


def main():
    args = parse_args()
    if args.op == "prepare":
        prepare(args)
    elif args.op == "train":
        train(args)
    elif args.op == "sample":
        sample(args)
    elif args.op == "recover":
        recover(args)
    else:
        raise ValueError(f"Operation {args.op} not recognized.")


if __name__ == "__main__":
    main()
