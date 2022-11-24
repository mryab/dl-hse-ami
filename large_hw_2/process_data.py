from argparse import ArgumentParser
from pathlib import Path

from data import convert_files, train_tokenizers

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "base_dir", type=Path, help="Path to the directory containing original data"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Path to the directory containing original data"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="The path that the trained tokenizer will be saved into",
    )
    args = parser.parse_args()
    
    convert_files(args.base_dir, args.output_dir)
    train_tokenizers(args.output_dir, args.tokenizer_path)
