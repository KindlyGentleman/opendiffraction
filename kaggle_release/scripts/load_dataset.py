import h5py
import numpy as np

def load_sample(h5_path, index=0):
    with h5py.File(h5_path, 'r') as f:
        inp = f['input'][index]
        tgt = f['target'][index]
        msk = f['mask'][index]
    return inp, tgt, msk

if __name__ == '__main__':
    print("OpenDiffraction standalone loader ready.")
