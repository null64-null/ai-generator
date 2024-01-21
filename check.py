import numpy as np

def output_size(input_size, filter_sizes, padding, stride):
    i_len = 1 + (input_size[0] + 2 * padding - filter_sizes[0]) / stride
    j_len = 1 + (input_size[1] + 2 * padding - filter_sizes[1]) / stride
    print(f"output size : ({i_len}, {j_len})")

input_size = [28, 28]
filter_size = [6, 6]
padding = 2
stride = 3

output_size(input_size, filter_size, padding, stride)