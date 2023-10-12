import sys

if __name__ == "__main__":
    if sys.version_info[0:2] != (3, 11):
        raise Exception("Requires python 3.11")