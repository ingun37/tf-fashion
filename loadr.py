from scipy.io import loadmat
import numpy as np
from PIL import Image


# def load_emnist_balanced_data() -> ((np.array, np.array), (np.array, np.array), list):
#     m = loadmat('emnist-byclass.mat')
#     data = m['dataset'][0, 0]

#     def decode_data(data):
#         num = data[0].shape[0]
#         datas = data[0].reshape((num, 28, 28, 1)).transpose([0, 2, 1, 3]).astype('uint8')
#         labels = data[1].astype('uint8')
#         return (datas, labels)

#     def class_names():
#         def compose(g, f):
#             '''Right to left function composition.'''
#             return lambda x: g(f(x))
#         str_chr = compose(str, chr)

#         num_names = map(str, range(10))
#         cap_names = map(str_chr, range(ord('A'), ord('Z')+1))
#         low_names = map(str_chr, range(ord('a'), ord('z')+1))
#         merged_names = []
#         sorted_low_names = sorted(set(low_names) - set(merged_names))

#         return list(num_names) + list(cap_names) + list(sorted_low_names)

#     return decode_data(data[0][0, 0]), decode_data(data[1][0, 0]), class_names()
def compose(g, f):
    '''Right to left function composition.'''
    return lambda x: g(f(x))
def digits_class_names():
    return list(map(str, range(10)))

def load_emnist_digits_data() -> ((np.array, np.array), (np.array, np.array), list):
    m = loadmat('emnist-digits.mat')
    data = m['dataset'][0, 0]

    def decode_data(data):
        num = data[0].shape[0]
        datas = data[0].reshape((num, 28, 28, 1)).transpose([0, 2, 1, 3]).astype('uint8')
        labels = data[1].astype('uint8')
        return (datas, labels)

    return decode_data(data[0][0, 0]), decode_data(data[1][0, 0]), digits_class_names()

