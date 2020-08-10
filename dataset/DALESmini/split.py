"""
This script will randomly split a directory of .ply files and place them in train, test, and val directories.
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
import numpy as np
import argparse


def main(in_path: str, out_path: str, val_size: float, test_size: float):
    in_dir = Path(in_path)
    out_dir = Path(out_path)
    
    train_dir = out_dir / 'train'
    test_dir = out_dir / 'test'
    val_dir = out_dir / 'val'

    scenes = [scene for scene in in_dir.iterdir() if scene.is_dir()]

    for scene in scenes:
        files = [f for f in scene.iterdir() if f.is_file()]
        names = list(set([f.name[:21] for f in files]))
        df = pd.DataFrame(data=names)
        train_names, test_names, val_names = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])
        
        for name in train_names[0]:
            batch_files = [bf for bf in files if bf.name.startswith(name)]
            for bf in batch_files:
                shutil.copy(bf, str(train_dir) + '/' + bf.name)
                print("Created {}".format(str(train_dir) + '/' + bf.name))

        for name in test_names[0]:
            batch_files = [bf for bf in files if bf.name.startswith(name)]
            for bf in batch_files:
                shutil.copy(bf, str(test_dir) + '/' + bf.name)
                print("Created {}".format(str(test_dir) + '/' + bf.name))

        for name in val_names[0]:
            batch_files = [bf for bf in files if bf.name.startswith(name)]
            for bf in batch_files:
                shutil.copy(bf, str(val_dir) + '/' + bf.name)
                print("Created {}".format(str(val_dir) + '/' + bf.name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split PointGroup compatible data into train/test/val.")
    parser.add_argument("-i", "--in_path", type=str, default="./ply_grid_8",
        help="Path to input directory containing aerial LiDAR .ply scenes.")
    parser.add_argument("-o", "--out_path", type=str, default="./",
        help="Path to output directory containing train/test/val directories.")
    parser.add_argument("-v", "--val_size", type=float, default="0.15",
        help="Proportional size of the validation set.")
    parser.add_argument("-t", "--test_size", type=float, default="0.15",
        help="Proportional size of the test set.")
    args = parser.parse_args()
    
    main(args.in_path, args.out_path, args.val_size, args.test_size)
