"""A small CLI with a flag-parsing bug."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="world")
    args = parser.parse_args(argv)
    # BUG: reads the wrong attribute name, crashing on --name.
    print(f"Hello, {args.username}!")


if __name__ == "__main__":
    main()
