import argparse
import json

from fight_data import write_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    summary = write_split(args.root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
