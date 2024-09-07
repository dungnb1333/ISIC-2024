# ISIC-2024

![Alt text](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4972760%2F169b1f691322233e7b31aabaf6716ff3%2Fex-tiles.png?generation=1717700538524806&alt=media "Optional Title")

4th place solution for [ISIC 2024 - Skin Cancer Detection Challenge](https://www.kaggle.com/competitions/isic-2024-challenge/overview)

### 1. Environment
- Ubuntu 22.04 LTS
- CUDA 12.1
- Nvidia Driver Version: 535.161.07
- Python 3.10.13
- GPU: 40GB, RAM: 64GB

```shell
conda create -n venv python=3.10.13
conda activate venv
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 2. Data preparation
- Download [competition dataset](https://www.kaggle.com/competitions/isic-2024-challenge/data) and extract to folder [./dataset/isic_2024](./dataset/isic_2024)
- Download [isic 2020 dataset](https://challenge.isic-archive.com/data/#2020) and extract to folder [./dataset/isic_2020](./dataset/isic_2020)
- Download [isic 2019 dataset](https://challenge.isic-archive.com/data/#2019) and extract to folder [./dataset/isic_2019](./dataset/isic_2019)
- Download [isic 2018 dataset](https://challenge.isic-archive.com/data/#2018) and extract to folder [./dataset/isic_2018](./dataset/isic_2018)
- Download [PAD UFES 20 dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1) and extract to folder [./dataset/PAD-UFES-20](./dataset/PAD-UFES-20)
- dataset structure should be [./dataset/dataset_structure.txt](./dataset/dataset_structure.txt)
- Then run following scripts
```shell
cd src
python split_2024.py
python split_2020.py
python split_2019.py
python split_2018.py
python split_pad_ufes_20.py
python prepare_isic_2024_tabular_v2.py
```
### 3. Train models
#### 3.1 Image pipeline
#### 3.1.1 Isic 2018 lesion segmentation
- To create masks for aux models (segmentation + multi-label classification), I trained 3 models with ISIC 2018 data
```
cd src/lesion_segmentation_isic_2018
python train.py --cfg configs/eb5_unet++.yaml
python train.py --cfg configs/eb7_unet++.yaml
python train.py --cfg configs/mit_b5_fpn.yaml
```
- Result

| Backbone        | Decoder | Image size | IoU   |
| :-------------- | :------ | :--------- | :---- |
| efficientnet-b5 | Unet++  | 512        | 0.827 |
| efficientnet-b7 | Unet++  | 256        | 0.829 |
| Mit B5          | FPN     | 512        | 0.843 |

- Then run following scripts
```
python predict_2024.py && python predict_2020.py && python predict_2019.py && python predict_pad_ufes.py
```

#### 3.1.2 Isic 2024 multi-label classification model
