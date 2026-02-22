import numpy as np
import csv
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser(description="Read single CSV row")
parser.add_argument("--csv",default=os.getenv('HOME')+"/Downloads/MNIST_CSV/mnist_test.csv", help="Path to CSV file")
parser.add_argument("--ri",type=int, help="Zero-based row index")
args = parser.parse_args()

def read_one_row(row_index, path, delimiter=','):
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i == row_index:
                return row
    raise IndexError("Row index out of range")

row = read_one_row(args.ri,args.csv)
row = row[1:]

img_array = np.array(row, dtype=np.uint8).reshape(28, 28)
img = Image.fromarray(img_array, mode="L")  # "L" = grayscale
img.show()



