"""
This script prints the number of each semantic label present in each
inst_nostuff.pth file created by prepare_data_inst_DALES.py.
"""

import glob
import torch
import numpy as np

if __name__ == "__main__":
    splits = ['train', 'val']
    for split in splits:
        files = sorted(glob.glob('{}/scene*_inst_nostuff.pth'.format(split)))
        rooms = [torch.load(i) for i in files]

        for i in range(len(rooms)):
            _, _, label, _ = rooms[i]   # label 0~19 -100;  instance_label 0~instance_num-1 -100
            scene_name = files[i].split('/')[-1][:21]
            print('{}/{} {}'.format(i + 1, len(rooms), scene_name))    
            semantic_label, counts = np.unique(label, return_counts=True)
            print("{}".format(dict(zip(semantic_label, counts))))
