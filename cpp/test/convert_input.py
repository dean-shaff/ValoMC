# Convert MC3Dmex.input.mat file into JSON file that is easier to read from C++ side.
import sys
import os
import json

import scipy.io


def convert_mat(file_path: str):
    base, ext = os.path.splitext(file_path)
    output_file_path = f"{base}.json"
    mat = scipy.io.loadmat(file_path)
    H_arr = mat["H"]

    output = {}
    for key in mat:
        if key.startswith("__"):
            continue
        arr = mat[key]
        output[key] = arr.tolist()

    with open(output_file_path, 'w') as fd:
        json.dump(output, fd)

# def verify_json(file_path: str):
#     with open(file_path, "r") as fd:
#         obj = json.load(fd)
#
#     H_arr = obj["H"]
#
#     for idx in range(10):
#         print(H_arr[idx])
#         # for idy in len(H_arr[0]):
#
if __name__ == "__main__":
    file_path = sys.argv[1]
    convert_mat(file_path)
    # verify_json(file_path)
