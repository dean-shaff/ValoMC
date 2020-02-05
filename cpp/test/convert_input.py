# Convert MC3Dmex.input.mat file into JSON file that is easier to read from C++ side.
import sys
import os
import json

import scipy.io


def convert_mat(file_path: str):
    base, ext = os.path.splitext(file_path)
    output_file_path = f"{base}.json"
    mat = scipy.io.loadmat(file_path)

    output = {}
    for key in mat:
        if key.startswith("__"):
            continue
        arr = mat[key]
        output[key] = arr.tolist()

    with open(output_file_path, 'w') as fd:
        json.dump(output, fd)


if __name__ == "__main__":
    file_path = sys.argv[1]
    convert_mat(file_path)
