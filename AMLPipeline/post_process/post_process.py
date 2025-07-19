import sys
import os
# Get the directory three levels up
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append it to sys.path
sys.path.append(parent_dir)

from datetime import datetime
import shutil
import argparse


def copy_data(input_dir, output_dir):
    # today's date
    today = datetime.now().strftime("%Y-%m-%d")

    # create the destination folder
    destination = os.path.join(output_dir, today)
    os.makedirs(destination, exist_ok=True)

    # copy the files
    for file in os.listdir(input_dir):
        s = os.path.join(input_dir, file)
        d = os.path.join(destination, file)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    print(f"Data copied from {input_dir} to {destination}")

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    return parser.parse_args()

def main(args):

    # copy folder
    copy_data(args.input_dir, args.output_dir)

if __name__ == '__main__':
    args = parse_parser()
    main(args)