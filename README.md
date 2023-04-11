# EMSAssist

This repository currently contains the reproducible artifact for EMSAssist.

## Using Docker

We follow the [official docker guide](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04) to install and run docker images:


## Using bare metal machine 
First of all, we download anaconda for smoother artifact evaluation

Download Anaconda installer: `wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh`

Run the installer: `bash Anaconda3-2023.03-Linux-x86_64.sh`. Keep pressing `Enter` or inputing `yes` on the command line

Create a conda environment for EMSAssist: `conda create -n emsassist-gpu pip python=3.7`

Activate the environment: `conda activate emsassist-gpu`

Install the XGBoost-GPU: `conda install py-xgboost-gpu`. This also installs the CudaToolkit: pkgs/main/linux-64::cudatoolkit-10.0.130-0 

Install the TensorFlow-2.9: `pip install tensorflow-gpu==2.9`

<!-- we create and activate a conda environment with tensorflow-gpu: `conda activate tf-gpu` -->

## Basic Environment

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 22.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |

| Hardware Environment  | Version |
| ------------- | ------------- |
| GPU  | 3 x NVIDIA A30   |
| CPU | 2 x Intel Xeon 4314 |

Before the artifact evaluation and use the open-sourced code/data, please make sure you have at least 1 NVIDIA GPU available with `nvidia-smi` command.

![nvidia-gpu](./nvidia-smi.png)

## Build the target Environment

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 22.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |
| CUDA Version  | 10   |
| TensorFlow  | 2.9   |


```
conda create -n xgb-gpu
conda activate xgb-gpu
conda install python=3.7
conda install py-xgboost-gpu
pip install tensorflow-gpu==2.9
```

`conda install -c conda-forge py-xgboost-gpu`

`mv /home/liuyi/anaconda3/lib/libstdc++.so.6.0.29 /home/liuyi/anaconda3/lib/libstdc++.so.6.0.29.old`

`ln -s /home/liuyi/anaconda3/envs/tf-gpu/lib/libstdc++.so.6.0.30 /home/liuyi/anaconda3/lib/libstdc++.so.6.0.29`