import sys
import config
import pre_processing as pre


def main():
    config.INPUT_PATH = sys.argv[1]
    config.OUTPUT_PATH = sys.argv[2]
    pre.build_baseline()


if __name__ == "__main__":
    main()
