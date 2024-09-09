# ISIC-2024

![Alt text](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4972760%2F169b1f691322233e7b31aabaf6716ff3%2Fex-tiles.png?generation=1717700538524806&alt=media "Optional Title")

4th place for [ISIC 2024 - Skin Cancer Detection Challenge](https://www.kaggle.com/competitions/isic-2024-challenge/overview)

### [Summary](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532760)

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
cd src/prepare
python split_2024.py
python split_2020.py
python split_2019.py
python split_2018.py
python split_pad_ufes_20.py
python prepare_isic_2024_tabular_v2.py
```
### 3. Train models
#### 3.1 Image pipeline

#### 3.1.1 Multi-label classification exp0
- 4 multi-label classification models trained with ISIC 2024+2020+2019 + PAD UFES, validated with ISIC 2024
```shell
cd src/image_exp0
### For Leaderboard Prize
python train.py --cfg configs/swin_tiny_224.yaml && python predict_oof.py --cfg configs/swin_tiny_224.yaml
python train.py --cfg configs/convnextv2_base_128.yaml && python predict_oof.py --cfg configs/convnextv2_base_128.yaml
python train.py --cfg configs/convnextv2_large_64.yaml && python predict_oof.py --cfg configs/convnextv2_large_64.yaml
python train.py --cfg configs/coatnet_rmlp_1_224.yaml && python predict_oof.py --cfg configs/coatnet_rmlp_1_224.yaml
python train_ema.py --cfg configs/swin_tiny_224.yaml
python train_ema.py --cfg configs/convnextv2_base_128.yaml
python train_ema.py --cfg configs/convnextv2_large_64.yaml
python train_ema.py --cfg configs/coatnet_rmlp_1_224.yaml

### For Secondary Prizes
python train.py --cfg configs/vit_tiny_224.yaml && python predict_oof.py --cfg configs/vit_tiny_224.yaml
python train_ema.py --cfg configs/vit_tiny_224.yaml
```
- Result

| Backbone         | Image size | CV pAUC |
| :--------------- | :--------- | :------ |
| [swin_tiny](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=swin_tiny_patch4_window7_224.ms_in1k_224_exp0)        | 224        | 0.1609  |
| [convnextv2_base](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=convnextv2_base.fcmae_ft_in22k_in1k_128_exp0)  | 128        | 0.1641  |
| [convnextv2_large](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=convnextv2_large.fcmae_ft_in22k_in1k_64_exp0) | 64         | 0.1642  |
| [coatnet_rmlp_1](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=coatnet_rmlp_1_rw_224.sw_in1k_224_exp0)   | 224        | 0.1617  |
| [vit_tiny](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=vit_tiny_patch16_224.augreg_in21k_ft_in1k_224_exp0)         | 224        | 0.1640  |

#### 3.1.2 Lesion segmentation
- To create masks for aux models (segmentation + multi-label classification) exp1, I trained 3 models with ISIC 2018 data
```shell
cd src/lesion_segmentation
python train.py --cfg configs/eb5_unet++.yaml
python train.py --cfg configs/eb7_unet++.yaml
python train.py --cfg configs/mit_b5_fpn.yaml
```
- Result

| Backbone        | Decoder | Image size | IoU   |
| :-------------- | :------ | :--------- | :---- |
| [efficientnet-b5](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=lesion_segmentation) | Unet++  | 512        | 0.827 |
| [efficientnet-b7](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=lesion_segmentation) | Unet++  | 256        | 0.829 |
| [mit-b5](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=lesion_segmentation)          | FPN     | 512        | 0.843 |

- Then run following scripts to create mask
```shell
python predict_2024.py && python predict_2020.py && python predict_2019.py && python predict_pad_ufes.py
```

#### 3.1.3 Multi-task classification + segmentation
- 3 multi-task segmentation + classification models trained with ISIC 2024+2020+2019 + PAD UFES, validated with ISIC 2024. For the submission, I only used the prediction from the classification task.
```shell
cd src/image_exp1_aux
### For Leaderboard Prize
python train.py --cfg configs/eb3_224.yaml && python predict_oof.py --cfg configs/eb3_224.yaml
python train.py --cfg configs/mit_b0_384.yaml && python predict_oof.py --cfg configs/mit_b0_384.yaml
python train.py --cfg configs/mit_b5_224.yaml && python predict_oof.py --cfg configs/mit_b5_224.yaml
python train_ema.py --cfg configs/eb3_224.yaml
python train_ema.py --cfg configs/mit_b0_384.yaml
python train_ema.py --cfg configs/mit_b5_224.yaml

### For Secondary Prizes
python train.py --cfg configs/mit_b0_224.yaml && python predict_oof.py --cfg configs/mit_b0_224.yaml
python train_ema.py --cfg configs/mit_b0_224.yaml
```
- Result

| Backbone        | Decoder | Image size | CV pAUC |
| :-------------- | :------ | :--------- | :------ |
| [efficientnet-b3](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=timm-efficientnet-b3_224_aux_exp1) | Unet    | 224        | 0.1638  |
| [mit-b0](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=mit_b0_384_aux_exp1)          | FPN     | 384        | 0.1671  |
| [mit-b5](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=mit_b5_224_aux_exp1)          | FPN     | 224        | 0.1656  |
| [mit-b0](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=mit_b0_224_aux_exp1)          | FPN     | 224        | 0.1660  |


#### 3.1.4 Multi-label classification exp3
- 3 multi-label classification models trained only with ISIC 2024 data
```shell
cd src/image_exp3
python train.py --cfg configs/vit_tiny_384.yaml && python predict_oof.py --cfg configs/vit_tiny_384.yaml
python train.py --cfg configs/swin_tiny_256.yaml && python predict_oof.py --cfg configs/swin_tiny_256.yaml
python train.py --cfg configs/convnextv2_tiny_288.yaml && python predict_oof.py --cfg configs/convnextv2_tiny_288.yaml
python train_ema.py --cfg configs/vit_tiny_384.yaml
python train_ema.py --cfg configs/swin_tiny_256.yaml
python train_ema.py --cfg configs/convnextv2_tiny_288.yaml
```
- Result

| Backbone        | Image size | CV pAUC |
| :-------------- | :--------- | :------ |
| [vit_tiny](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=vit_tiny_patch16_384.augreg_in21k_ft_in1k_384_exp3)        | 384        | 0.1688  |
| [swin_tiny](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=swinv2_tiny_window8_256.ms_in1k_256_exp3)       | 256        | 0.1655  |
| [convnextv2_tiny](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=convnextv2_tiny.fcmae_ft_in22k_in1k_288_exp3) | 288        | 0.1645  |

#### 3.2 Tabular pipeline
I used 3 models: LightGBM, CatBoost, XGBoost. I combined 10 features from the image pipeline and all the features from the [amazing tabular notebook](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data/notebook)
```shell
cd src/tabular
python train_tab_meta_feat.py
### For Leaderboard Prize
python train_10_model_feat.py
### For Secondary Prizes
python train_2_model_feat.py
```
- Result

|                  | LGB pAUC | CB pAUC | XGB pAUC | Gmean pAUC |
| :--------------- | :------- | :------ | :------- | :--------- |
| [meta feat](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=tab_meta_feat)    | 0.17806  | 0.17498 | 0.17954  | 0.17879    |
| [10model feat](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=tab_10model_feat) | 0.18650  | 0.18669 | 0.18647  | 0.18703    | 
| [2model feat](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=tab_2model_feat)  | 0.18406  | 0.18378 | 0.18328  | 0.18438    | 


### 4.FINAL SUBMISSION
|                                                    | PublicLB | PrivateLB |
| :------------------------------------------------- | :------- | :-------- |
| [Sub1-GPU: 0.2*(meta only) + 0.8*(10model feat)](https://www.kaggle.com/code/nguyenbadung/isic-2024-final-submission?scriptVersionId=195231736) | 0.182    | 0.172     |
| [Sub2-CPU: 0.2*(meta only) + 0.8*(2model feat)](https://www.kaggle.com/code/nguyenbadung/isic-2024-secondary-prize?scriptVersionId=195319448)  | 0.180    | 0.170     |
