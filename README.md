<!-- PROJECT LOGO -->

<div align="center">
<h1>DIFNet</h1>
<h3>DIFNet: Frequency-aware Processing of High and Low-frequency Components for Depth Completion</h3>

[Kunyang Wu](https://scholar.google.com/citations?hl=en&user=W2pNdNEAAAAJ)<sup>1,3</sup>, [Jun Lin](https://ciee.jlu.edu.cn/info/1162/8474.htm)<sup>1,2,3</sup>, [Jiawei Miao](https://ieeexplore.ieee.org/author/268567427197560)<sup>4</sup>, [Zhengpeng Li](https://ieeexplore.ieee.org/author/37086228429)<sup>4</sup>, [Xiucai Zhang](https://ieeexplore.ieee.org/author/222732598856523)<sup>1</sup>, [Genyuan Xing](https://scholar.google.com/citations?user=O-ld4UUAAAAJ&hl=en)<sup>1</sup>, [Yiyao Fan](https://ieeexplore.ieee.org/author/142426428990897)<sup>1</sup>, [Jinxin Luo](https://ieeexplore.ieee.org/author/37089468294)<sup>1</sup>, [Huanyu Zhao](https://ciee.jlu.edu.cn/info/1156/14433.htm)<sup>1</sup>, [Yang Liu](https://scholar.google.com/citations?user=rN6ryXIAAAAJ)<sup>1,3*</sup>, [Guanyu Zhang](https://scholar.google.com/citations?user=mJb59RcAAAAJ&hl=en)<sup>1,3*</sup>

<sup>1</sup> Jilin University, <sup>2</sup> National Key Laboratory of Deep Exploration and Imaging, <sup>3</sup> Institute of Intelligent Instruments and Measurement Controlling Technology, <sup>4</sup> University of Science and Technology Liaoning

<sup>*</sup> Corresponding authors: liu_yang@jlu.edu.cn, zhangguanyu@jlu.edu.cn

</div>

## ✅ Main Results

### **Depth Completion on KITTI Benchmark**

| Method | RMSE (mm) ↓ | MAE (mm) ↓ | iRMSE (1/km) ↓ | iMAE (1/km) ↓ | FLOPs (G) ↓ | Reference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NLSPN | 741.68 | 199.59 | 1.99 | **0.84** | 492.72 | ECCV 2020 |
| MDANet | 738.23 | 214.99 | 2.12 | 0.99 | 225.39 | ICRA 2021 |
| PENet | 730.08 | 210.55 | 2.17 | 0.94 | 405.82 | ICRA 2021 |
| GuideNet | 736.24 | 218.83 | 2.25 | 0.99 | 229.95 | TIP 2021 |
| ACMNet | 744.91 | 206.09 | 2.08 | 0.90 | 272.57 | TIP 2021 |
| UnDCMaster | 751.59 | *198.09* | 1.98 | *0.85* | — | AAAI 2022 |
| ReDC | 728.31 | 204.60 | 2.05 | 0.89 | 381.23 | IROS 2023 |
| PointDC | 736.07 | 201.87 | *1.97* | 0.87 | 489.05 | ICCV 2023 |
| CFormer | **708.87** | 203.45 | 2.01 | 0.88 | 435.33 | CVPR 2023 |
| CluDe | 734.59 | 200.48 | 2.08 | 0.88 | — | TCSVT 2024 |
| CHNet | 734.83 | 213.48 | 2.23 | 0.95 | 177.68 | KBS 2024 |
| GeoDC | 736.06 | 213.02 | 2.17 | 0.95 | 292.73 | TGRS 2024 |
| SCMT | *719.65* | 208.03 | 2.02 | 0.89 | — | TIP 2024 |
| DIFNet (Ours) | 720.22 | **195.23** | **1.95** | *0.85* | 352.45 | — |

* *Models in this subsection are evaluated on the KITTI Depth Completion test set. **Bold** indicates best performance, while *italic* indicates second-best performance.*
* *DIFNet introduces a sophisticated early fusion layer and explicit frequency-aware processing, balancing efficiency and precision with competitive FLOPs (352.45G) compared to other methods.*
* *While CFormer achieves the best RMSE (708.87mm) through its hybrid ResNet-transformer architecture, our approach excels in MAE and iRMSE metrics, demonstrating superior overall performance.*

### **Depth Completion on NYUv2 Benchmark**

| Method | RMSE (m) ↓ | REL ↓ | δ1.25 ↑ | Reference |
| :---: | :---: | :---: | :---: | :---: |
| NLSPN | 0.092 | **0.012** | 0.995 | ECCV 2020 |
| GuideNet | 0.101 | 0.015 | 0.995 | TIP 2021 |
| ACMNet | 0.105 | 0.015 | 0.995 | TIP 2021 |
| AGGNet | 0.092 | **0.012** | 0.994 | ICCV 2021 |
| PointDC | **0.089** | **0.012** | 0.996 | ICCV 2023 |
| CFormer | *0.090* | **0.012** | 0.996 | CVPR 2023 |
| CHNet | 0.099 | 0.016 | 0.995 | KBS 2024 |
| GeoDC | *0.090* | **0.012** | 0.996 | TGRS 2024 |
| SCMT | 0.092 | *0.013* | 0.996 | TIP 2024 |
| DIFNet (Ours) | **0.089** | **0.012** | 0.996 | — |

* *Models in this subsection are evaluated on the NYUv2 indoor depth completion dataset. **Bold** indicates best performance, while *italic* indicates second-best performance.*
* *Our DIFNet achieves state-of-the-art performance, matching PointDC with the best RMSE (0.089m) while also obtaining optimal REL (0.012) and δ1.25 (0.996) metrics.*


## ⚙️ Installation

### Step 1: Clone the DIFNet repository:

To get started, first clone the DIFNet repository and navigate to the project directory:

```bash
git clone https://github.com/wuky2000/DIFNet.git
cd DIFNet
```

### Step 2: Environment Setup:

DINet recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
Also, We recommend using the pytorch>=2.0, cuda>=11.8.

**Create and activate a new conda environment**

```bash
conda create -n difnet python=3.10
conda activate difnet
```

**Install Vmamba**

```bash
conda install cudatoolkit==11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install setuptools==68.2.2
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

**Install Dependencies**

```bash
pip install mmcv-full==1.4.4 mmsegmentation==0.22.1  
pip install timm tqdm thop tensorboardX opencv-python ipdb h5py ipython Pillow==9.5.0 
```

**NVIDIA Apex**

We used NVIDIA Apex (commit @ 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a) for multi-GPU training.

Apex can be installed as follows:

```bash
$ cd PATH_TO_INSTALL
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```


## 💾 Datasets
We used two datasets for training and evaluation.

### NYU Depth V2 (NYUv2)

We used preprocessed NYUv2 HDF5 dataset provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).

```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```

After that, you will get a data structure as follows:

```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

Note that the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

After preparing the dataset, you should generate a json file containing paths to individual images.

```bash
$ cd THIS_PROJECT_ROOT/utils
$ python generate_json_NYUDepthV2.py --path_root PATH_TO_NYUv2
```

Note that data lists for NYUv2 are borrowed from the [CSPN repository](https://github.com/XinJCheng/CSPN/tree/master/cspn_pytorch/datalist).


### KITTI Depth Completion (KITTI DC)

KITTI DC dataset is available at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php). You can refer to [this script](https://github.com/youmi-zym/CompletionFormer/issues/8#issuecomment-1602302424) for data preparation.

The overall data directory is structured as follows:

```
├── kitti_depth
|   ├──data_depth_annotated
|   |  ├── train
|   |  ├── val
|   ├── data_depth_velodyne
|   |  ├── train
|   |  ├── val
|   ├── data_depth_selection
|   |  ├── test_depth_completion_anonymous
|   |  |── test_depth_prediction_anonymous
|   |  ├── val_selection_cropped
|   ├── raw_data
|   |   ├── 2011_09_26
|   |   ├── 2011_09_28
|   |   ├── 2011_09_29
|   |   ├── 2011_09_30
|   |   ├── 2011_10_03
```

After preparing the dataset, you should generate a json file containing paths to individual images. 

```bash
$ cd THIS_PROJECT_ROOT/utils

# For Train / Validation
$ python generate_json_KITTI_DC.py --path_root PATH_TO_KITTI_DC

# For Online Evaluation Data
$ python generate_json_KITTI_DC.py --path_root PATH_TO_KITTI_DC --name_out kitti_dc_test.json --test_data
```


## ⏳ Model Training

Note: batch size is set for each GPU

```bash
$ cd THIS_PROJECT_ROOT/src

# An example command for NYUv2 dataset training
$ python main.py --dir_data PATH_TO_NYUv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --loss 1.0*L1+1.0*L2 --batch_size 48 --milestones 36 48 56 64 72 --epochs 72 \
    --log_dir ../experiments/ --save NAME_TO_SAVE \
    
# An example command for KITTI DC dataset training: L2 loss
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --gpus 0,1,2,3 --loss 1.0*L1 --lidar_lines 64 \
    --batch_size 12 --max_depth 90.0 --lr 0.001 --epochs 100 --milestones 50 60 70 80 90 100 \
    --top_crop 100 --test_crop --log_dir ../experiments/ --save NAME_TO_SAVE \

# An example command for KITTI DC dataset training: L1 + L2 loss
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --gpus 0,1,2,3 --loss 1.0*L1+1.0*L2 --lidar_lines 64 \
    --batch_size 12 --max_depth 90.0 --lr 0.001 --epochs 100 --milestones 50 60 70 80 90 100 \
    --top_crop 100 --test_crop --log_dir ../experiments/ --save NAME_TO_SAVE \
```


## 📊 Testing

```bash
$ cd THIS_PROJECT_ROOT/src

# An example command for NYUv2 dataset testing
$ python main.py --dir_data PATH_TO_NYUv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
    --test_only --pretrain PATH_TO_WEIGHTS --save NAME_TO_SAVE

# An example command for KITTI DC dataset testing
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 --top_crop 100 --test_crop --save_image \
    --test_only --pretrain PATH_TO_WEIGHTS --save NAME_TO_SAVE
```

**Pretrained Checkpoints**: [NYUv2](https://drive.google.com/file/d/1HN9lFwEMMFtAJ0tIwnoNde0ONDDDb_yz/view?usp=sharing), [KITTI_DC](https://drive.google.com/file/d/13giNyJGJ4LeePyE2YwzeO9kO6G6ucZY_/view?usp=sharing)!

To generate KITTI DC Online evaluation data:

```bash
$ cd THIS_PROJECT_ROOT/src
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dc_test.json \
    --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 \
    --test_only --pretrain PATH_TO_WEIGHTS --save_image --save_result_only --save NAME_TO_SAVE
```

Images for submission can be found in THIS_PROJECT_ROOT/experiments/NAME_TO_SAVE/test/epoch%04d.


## 👩‍⚖️ Acknowledgement

Thanks the authors for their works: [NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20), [Completionformer](https://github.com/youmi-zym/CompletionFormer), [Vmamba](https://github.com/MzeroMiko/VMamba).

Besides, we also thank [CHNet](https://github.com/lmomoy/CHNet/issues) for providing their pre-training weights on KITTI DC.
