import sys
import pre_processing as pre
from config import *

def main():
    pre.build_baseline()


if __name__ == "__main__":
    print(sys.argv)
    INPUT_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    print(INPUT_PATH, OUTPUT_PATH)
    main()

