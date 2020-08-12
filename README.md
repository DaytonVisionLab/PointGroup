# PointGroup
## PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation (CVPR2020)
![overview](https://github.com/llijiang/PointGroup/blob/master/doc/overview.png)

Code for the paper **PointGroup:Dual-Set Point Grouping for 3D Instance Segmentation**, CVPR 2020 (Oral).

**Authors**: Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, Jiaya Jia 

[[arxiv]](https://arxiv.org/abs/2004.01658) [[video]](https://youtu.be/HMetye3gmAs)

## Introduction
Instance segmentation is an important task for scene understanding. Compared to the fully-developed 2D, 3D instance segmentation for point clouds have much room to improve. In this paper, we present PointGroup, a new end-to-end bottom-up architecture, specifically focused on better grouping the points by exploring the void space between objects. We design a two-branch network to extract point features and predict semantic labels and offsets, for shifting each point towards its respective instance centroid. A clustering component is followed to utilize both the original and offset-shifted point coordinate sets, taking advantage of their complementary strength. Further, we formulate the ScoreNet to evaluate the candidate instances, followed by the Non-Maximum Suppression (NMS) to remove duplicates.

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.1.0
* CUDA 10.0 (VisionLab uses RTX GPU which is incompatible with CUDA 9.0, we will need to use CUDA 10.0 instead of 9.0)

### Virtual Environment
```
conda create -n PointGroup python==3.7
source activate PointGroup
```

### Install CUDA
```
mkdir /home/vlab/anaconda3/envs/PointGroup/cuda
cd /home/vlab/anaconda3/envs/PointGroup/cuda
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod +x cuda_10.0.120_410.48_linux
./cuda_10.0.120_410.48_linux
```
* Accept the EULA. Hold spacebar to read and understand the EULA really, really fast.
* Do NOT install the driver. Install only the cudatoolkit.
* Set the cudatoolkit location to /home/vlab/anaconda3/envs/PointGroup/cuda
* Do NOT create a symbolic link in /usr/local.

### Install cuDNN
* Go to https://developer.nvidia.com/rdp/cudnn-download and select 'Download cuDNN v7.6.5 for CUDA 10.0'. (You will need to create/sign in to Nvidia account)
* Select 'cuDNN Library for Linux'.
* Download cudnn-10.0-linux-x64-v7.6.5.32.tgz and place it in /home/vlab/anaconda3/envs/PointGroup.
* Uncompress the file:
```
cd /home/vlab/anaconda3/envs/PointGroup
tar -xzvf cudnn-10.0-linux-x64-v7.6.5.32.tgz
```

### Set Environment Variables
```
export PATH=/home/vlab/anaconda3/envs/PointGroup/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/vlab/anaconda3/envs/PointGroup/cuda/lib64:$LD_LIBRARY_PATH
export CUDNN_LIBRARY=/home/vlab/anaconda3/envs/PointGroup/cuda/lib64/libcudnn.so
export CUDNN_INCLUDE_DIR=/home/vlab/anaconda3/envs/PointGroup/cuda/include
```

### Install `PointGroup`

(1) Clone the PointGroup repository.
```
git clone https://github.com/llijiang/PointGroup.git --recursive
cd PointGroup
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv). The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** We further modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use our modified `spconv`.

* To compile `spconv`, firstly install the dependent libraries. If the gcc-5 installation fails, see the note below.
```
conda install libboost
conda install -c conda-forge mpfr=4.0.2
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```

* The following error may occur: 
- error while loading shared libraries: libmpfr.so.4: cannot open shared object file: No such file or directory
- The error can be resolved by creating a symbolic link between libmpfr.so.4 and libmpfr.so.6:
```
ln -s /home/vlab/anaconda3/envs/pg3/lib/libmpfr.so.6 /home/vlab/anaconda3/envs/pg3/lib/libmpfr.so.4
```

Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.


# ScanNetV2

## Data Preparation

(1) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
* Create download_scannet.py in /PointGroup/dataset/scannetv2/ using the code found here: http://kaldir.vc.in.tum.de/scannet/download-scannet.py
* Replace all occurences of "raw_input" with "input" to work with python3.
* Uncomment the lines which reference "urllib.request.urlopen", and comment the lines which reference "urllib.urlopen" to work with python3.
* Use download_scannet.py to download several scans into /home/vlab/PointGroup/dataset/scannetv2/. 
* Training error will occur if <2 scans are used for training or validation data.
* Download the ScanNet label map.
- To download the entire ScanNet release (1.3TB): 
```
python download-scannet.py -o /home/vlab/PointGroup/dataset/scannetv2/
```
- To download specific scans (e.g., scene0000_00, scene0000_01, scene0000_03, scene0001_00, scene0002_00): 
```
python download-scannet.py -o /home/vlab/PointGroup/dataset/scannetv2/ --id scene0000_00
```
- To download a specific file type (e.g., *.sens, valid file suffixes listed here):
```
python download-scannet.py -o /home/vlab/PointGroup/dataset/scannetv2/ --type .sens
```
- To download the ScanNet v1 task data (inc. trained models): 
```
python download-scannet.py -o /home/vlab/PointGroup/dataset/scannetv2/ --task_data
```
- To download the ScanNet label map: 
```
python download-scannet.py -o /home/vlab/PointGroup/dataset/scannetv2/ --label_map
```


(2) Put the data in the corresponding folders. 
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.labels.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`  and `[scene_id].aggregation.json`  into the `dataset/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Copy the files `[scene_id]_vh_clean_2.ply` into the `dataset/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Put the file `scannetv2-labels.combined.tsv` in the `dataset/scannetv2` folder.

The dataset files are organized as follows.
```
PointGroup
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```

(3) Generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation.
```
cd dataset/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml 
```
You can start a tensorboard session by
```
tensorboard --logdir=./exp --port=6666
```

## Inference and Evaluation

(1) If you want to evaluate on validation set, prepare the `.txt` instance ground-truth files as the following.
```
cd dataset/scannetv2
python prepare_data_inst_gttxt.py
```
Make sure that you have prepared the `[scene_id]_inst_nostuff.pth` files before. 

(2) Test and evaluate. 

a. To evaluate on validation set, set `split` and `eval` in the config file as `val` and `True`. Then run 
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_run1_scannet.yaml
```
An alternative evaluation method is to set `save_instance` as `True`, and evaluate with the ScanNet official [evaluation script](https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py).

b. To run on test set, set (`split`, `eval`, `save_instance`) as (`test`, `False`, `True`). Then run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_run1_scannet.yaml
```

c. To test with a pretrained model, run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_default_scannet.yaml --pretrain $PATH_TO_PRETRAIN_MODEL$
```

## Pretrained Model
We provide a pretrained model trained on ScanNet v2 dataset. Download it [here](https://drive.google.com/file/d/1wGolvj73i-vNtvsHhg_KXonNH2eB_6-w/view?usp=sharing). Its performance on ScanNet v2 validation set is 35.2/57.1/71.4 in terms of mAP/mAP50/mAP25.


## Visualize
To visualize the point cloud, you should first install [mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html). Then you could visualize by running
```
cd util 
python visualize.py --data_root $DATA_ROOT$ --result_root $RESULT_ROOT$ --room_name $ROOM_NAME$ --room_split $ROOM_SPLIT$ --task $TASK$
```
The visualization task could be `input`, `instance_gt`, `instance_pred`, `semantic_pred` and `semantic_gt`.

## Results on ScanNet Benchmark 
Quantitative results on ScanNet test set at the submisison time.
![scannet_result](https://github.com/llijiang/PointGroup/blob/master/doc/scannet_benchmark.png)

# DALES

## Data Preparation

There are two version of the DALES dataset. One stores instances per scene as .txt files, and the other stores all of the scenes as .ply files. Both are available on the vision lab watercooled machine in /home/vlab/Desktop/DALES_data. 

Alternatively, the .txt data can be downloaded from:
```
https://drive.google.com/uc?id=1NK-sJOswWkkyp5OqQ4-HusidkyadXXiG&export=download
```
Unfortunately, this copy of the DALES dataset is missing ground instances for several scenes. These missing instances cause errors during training and testing, and so they must be removed. 

The .ply data can be downloaded from: 
```
https://drive.google.com/uc?id=1m1-W04ikcNLXfQfEgGZaGwuVnzXo4m_c&export=download
``` 
It should be noted that the network cannot handle these .ply files as-is. A new gridding method will likely be needed to train/test with these files.

## Data Preparation: DALES .ply

If you plan on using the DALES .ply files, make train, test, and val directories, then copy the training and testing files into them.
```
mkdir /dataset/DALES/train
mkdir /dataset/DALES/test
mkdir /dataset/DALES/val
cp /PATHTODATA/training_points /dataset/DALES/train
cp /PATHTODATA/test_points/{5080_54400.ply, 5100_54440.ply, 5120_54445.ply, 5135_54435.ply, 5150_54325.ply, 5175_54395.ply} /dataset/DALES/test
cp /PATHTODATA/test_points/{5080_54470.ply, 5100_54490.ply, 5135_54430.ply, 5140_54390.ply, 5155_54335.ply} /dataset/DALES/test
```

Next, create the pytorch instance files: 
```
python /dataset/DALES/prepare_data_inst_DALES.py --data_split train
python /dataset/DALES/prepare_data_inst_DALES.py --data_split test
python /dataset/DALES/prepare_data_inst_DALES.py --data_split val
```

Now you can generate the ground truth validation files.
```
cd /dataset/DALEStext
python prepare_data_inst_gttxt_DALES.py
```

To succesfully train on this data, experiment with network parameters (I specifically recommend trying different clustering parameters), or implement your own gridding method. If you decide to implement your own gridding method I recommend you look at the data prep script /data/DALES_inst.py and the old gridding method /dataset/DALEStext/txt_to_ply_gridding.py.

## Data Preparation: DALES .txt

Due to the aforementioned error with the .txt dataset, you will need to remove problematic files. For instance, certain files are missing their ground points. Files that DO have their ground instances can be identified using the command:
```
ls /PATH/TO/TXT/FILES/*/Annotations/ground*
```

Next, make train, test, val, and ply directories.
```
mkdir /dataset/DALEStext/train
mkdir /dataset/DALEStext/test
mkdir /dataset/DALEStext/val
mkdir /dataset/DALEStext/ply_grid_8
```

Now create .ply files using the conversion script. 
```
python /dataset/DALEStext/txt_to_ply_gridding.py -i /PATH/TO/FILTERED/TXT/FILES/ -o /dataset/DALEStext/ply_grid_8
```

Next, create the pytorch instance files: 
```
python /dataset/DALEStext/prepare_data_inst_DALES.py --data_split train
python /dataset/DALEStext/prepare_data_inst_DALES.py --data_split test
python /dataset/DALEStext/prepare_data_inst_DALES.py --data_split val
```

Now you can generate the ground truth validation files.
```
cd /dataset/DALEStext
python prepare_data_inst_gttxt_DALES.py
```

## Training

To train with the DALES data you've prepared, you can run the following command with the default DALES config file.
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_DALES.yaml 
```
You can start a tensorboard session by
```
tensorboard --logdir=./exp --port=6666
```

Similarly, you can train with the DALEStext data using the command:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_DALEStext.yaml 
```
You can start a tensorboard session by
```
tensorboard --logdir=./exp --port=6666
```

To experiment with different paramters, create a new config file and edit the parameter values. Trained models and predicted results will be stored for each config file separately.
```
cp /config/pointgroup_DALES.yaml /config/pointgroup_NEWPARAMS_DALES.yaml
```

## Inference and Evaluation

a. To evaluate on validation set, set `split` and `eval` in the desired config file as `val` and `True`. Then run 
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_DALES.yaml
```
An alternative evaluation method is to set `save_instance` as `True`, and then follow the visualization steps below. 

b. To run on test set, set (`split`, `eval`, `save_instance`) as (`test`, `False`, `True`). Then run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_DALES.yaml
```

c. To test with a specific pretrained model, run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_DALES.yaml --pretrain $PATH_TO_PRETRAIN_MODEL$
```

## Visualize 
To visualize the point cloud, you should first install CloudCompare. Then you must create CloudCompare compatible .txt files using the create_CC.py module in /PointGroup/utils. To create semantic segmentation results, run the following command: 
```
cd util 
python create_CC.py --data_root ../dataset/DALES --data_split test --result_root ../exp/DALES/pointgroup/pointgroup_r05_DALES/result/epoch256_nmst0.3_scoret0.09_npointt100/ --scene_name scene5080_54400_00_00
```
To create instance segmentation results, change the pred_type parameter like so:
```
cd util 
python create_CC.py --data_root ../dataset/DALES --data_split test --result_root ../exp/DALES/pointgroup/pointgroup_r05_DALES/result/epoch256_nmst0.3_scoret0.09_npointt100/ --scene_name scene5080_54400_00_00 --pred_type semantic
```

Now open the created file(s) with CloudCompare. The points should have been assigned RGB color values through the previous scripts, but this is currently untested. As an alternative, you can create a custom color scale for labels 1-8 and assign it to the semantic label scalar of the point cloud. 


# Citation
If you find this work useful in your research, please cite:
```
@article{jiang2020pointgroup,
  title={PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation},
  author={Jiang, Li and Zhao, Hengshuang and Shi, Shaoshuai and Liu, Shu and Fu, Chi-Wing and Jia, Jiaya},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Acknowledgement
This repo is built upon several repos, e.g., [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), [spconv](https://github.com/traveller59/spconv) and [ScanNet](https://github.com/ScanNet/ScanNet). 

## Contact
If you have any questions about this repo, please don't hesitate to email me (carrjp21@gmail.com). The original creator of PointGroup is also available (lijiang@cse.cuhk.edu.hk).


