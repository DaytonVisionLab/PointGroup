import numpy as np
import argparse
import torch
from pathlib import Path


def get_rgb_colors(preds):
    rgb = np.zeros((preds.shape[0],3), dtype=int)
    rgb[np.where(preds == 0)[0],:] = [0,0,0]        #unlabeled
    rgb[np.where(preds == 1)[0],:] = [0,0,255]      #ground
    rgb[np.where(preds == 2)[0],:] = [0,128,128]    #vegetation
    rgb[np.where(preds == 3)[0],:] = [0,128,0]      #cars
    rgb[np.where(preds == 4)[0],:] = [0,128,0]      #trucks
    rgb[np.where(preds == 5)[0],:] = [0,255,0]      #powerlines
    rgb[np.where(preds == 6)[0],:] = [255,128,0]    #fences
    rgb[np.where(preds == 7)[0],:] = [0,255,255]    #poles
    rgb[np.where(preds == 8)[0],:] = [255,0,0]      #buildings
    return rgb

def create_instance(opt):
    input_file = Path(opt.data_root, opt.data_split, (opt.scene_name + "_inst_nostuff.pth"))
    results_dir = Path(opt.result_root, opt.data_split)
    pred_file = results_dir / (opt.scene_name + ".txt")
    pred_CC = results_dir / (pred_file.stem + "_CC.txt")

    #Get xyz, ignore the rgb output so we can use our own colors later
    xyz, _ = torch.load(input_file)
    semantic_labels = np.zeros(np.shape(xyz)[0], dtype=int)
    instance_labels = np.ones(np.shape(xyz)[0], dtype=int) * -1

    semantic_pred_counts = {}

    with pred_file.open(mode="r") as preds:
        # preds is txt file where each line represents an instance pred.
        # Each pred is a string with format: "predicted_masks/scene5080_54400_0.txt semantic_label confidence"
        instances = [pred.split() for pred in preds]

        for instance in instances:
            mask_path = results_dir / Path(instance[0])
            semantic_label = int(instance[1])
            instance_label = int(mask_path.stem[-3:])
            mask = np.loadtxt(mask_path).astype(np.bool)
            semantic_labels[mask] = semantic_label
            instance_labels[mask] = instance_label

            #Track the number of instances per semantic label
            if semantic_label in semantic_pred_counts:
                semantic_pred_counts[semantic_label] += 1
            else:
                semantic_pred_counts[semantic_label] = 1

        rgb = get_rgb_colors(semantic_labels)

        with pred_CC.open(mode="w", encoding="utf-8") as cc:
            semantic_labels = np.expand_dims(semantic_labels, axis=1)
            instance_labels = np.expand_dims(instance_labels, axis=1)
            np.savetxt(cc, np.asarray(np.hstack((xyz, rgb, semantic_labels, instance_labels))))

    print("Creating: {}".format(pred_CC))
    print("Semantic label prediction counts: {}".format(semantic_pred_counts))

def create_semantic(opt):
    input_file = Path(opt.data_root, opt.data_split, (opt.scene_name + "_inst_nostuff.pth"))
    pred_file = Path(opt.result_root, opt.data_split, "semantic", (opt.scene_name + ".npy"))
    pred_CC = pred_file.parent / (pred_file.stem + "_CC.txt")

    #Get xyz, ignore the rgb output so we can use our own colors
    xyz, _ = torch.load(input_file)
    semantic_labels = np.load(pred_file)
    rgb = get_rgb_colors(semantic_labels)

    with pred_CC.open(mode="w", encoding="utf-8") as cc:
        semantic_labels = np.expand_dims(semantic_labels, axis=1)
        np.savetxt(cc, np.asarray(np.hstack((xyz, rgb, semantic_labels))))

    print("Creating: {}".format(pred_CC))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="path to the input dataset files", default="../dataset/DALES")
    parser.add_argument("--data_split", help="val / test", default="test")
    parser.add_argument("--result_root", help="path to the predicted results", default="../exp/DALES/pointgroup/pointgroup_r05_DALES/result/epoch256_nmst0.3_scoret0.09_npointt100/")
    parser.add_argument("--scene_name", help="name of scene e.g. scene5110_54320_02_05", default="scene5110_54320_02_05")
    parser.add_argument("--pred_type", help="semantic / instance", default="instance")
    opt = parser.parse_args()

    if opt.pred_type == "instance":
        create_instance(opt)
    elif opt.pred_type == "semantic":
        create_semantic(opt)
