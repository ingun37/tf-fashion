from scipy.io import loadmat
import numpy as np

def class_names():
    def compose(g, f):
        '''Right to left function composition.'''
        return lambda x: g(f(x))
    str_chr = compose(str, chr)

    num_names = map(str, range(10))
    cap_names = map(str_chr, range(ord('A'), ord('Z')+1))
    low_names = map(str_chr, range(ord('a'), ord('z')+1))
    merged_names = ['c','i','j','k','l','m','o','p','s','u','v','w','x','y','z']
    sorted_low_names = sorted(set(low_names) - set(merged_names))

    return list(num_names) + list(cap_names) + list(sorted_low_names)

def load_emnist_balanced_data() -> ((np.array, np.array), (np.array, np.array), list):
    m = loadmat('emnist-balanced.mat')
    data = m['dataset'][0, 0]

    def decode_data(data):
        num = data[0].shape[0]
        datas = data[0].reshape((num, 28, 28)).transpose([0,2,1]).astype('uint8')
        labels = data[1].astype('uint8')
        return (datas, labels)

    return decode_data(data[0][0, 0]), decode_data(data[1][0, 0]), class_names()
