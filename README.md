# Zero-shot face recognition: Improving the discriminability of visual face features using a Semantic-Guided Attention Model

This is the official implementation of the paper [Zero-shot face recognition: Improving the discriminability of visual face features using a Semantic-Guided Attention Model](https://www.sciencedirect.com/science/article/pii/S0957417422016803).

## Abstract

Zero-shot learning enables the recognition of classes not seen during training through the use of semantic information comprising a visual description of the class either in textual or attribute form. Despite the advances in the performance of zero-shot learning methods, most of the works do not explicitly exploit the correlation between the visual attributes of the image and their corresponding semantic attributes for learning discriminative visual features. In this paper, we introduce an attention-based strategy for deriving features from the image regions regarding the most prominent attributes of the image class. In particular, we train a Convolutional Neural Network (CNN) for image attribute prediction and use a gradient-weighted method for deriving the attention activation maps of the most salient image attributes. These maps are then incorporated into the feature extraction process of Zero-Shot Learning (ZSL) approaches for improving the discriminability of the features produced through the implicit inclusion of semantic information. For experimental validation, the performance of state-of-the-art ZSL methods was determined using features with and without the proposed attention model. Surprisingly, we discover that the proposed strategy degrades the performance of ZSL methods in classical ZSL datasets (AWA2), but it can significantly improve performance when using face datasets. Our experiments show that these results are a consequence of the interpretability of the dataset attributes, suggesting that existing ZSL datasets attributes are, in most cases, difficult to be identifiable in the image. Source code is available at https://github.com/CristianoPatricio/SGAM.

<p align="center"><img src="https://github.com/CristianoPatricio/SGAM/blob/main/figures/sgam_model.jpg" width="600"></p>

If you use this repository, please cite:

```
@article{
  author = {Cristiano Patrício and João Neves}
  title = {Zero-shot face recognition: Improving the discriminability of visual face features using a Semantic-Guided Attention Model},
  journal = {Expert Systems with Applications},
  pages = {118635},
  year = {2022},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2022.118635},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417422016803}
}
```

---

## 1. Data

| Dataset | Link | Pretrained Models |
| ----------- | ----------- | ------------- |
| LFWA | [LFWA Dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams?resourcekey=0-Kpdd6Vctf-AdJYfS55VULA&usp=sharing) | [Pretrained models](https://socia-lab.di.ubi.pt/~cristiano_patricio/data/pretrained_models_LFWA.zip) |

## 2. Training

First of all, create a new conda environment with the required libraries contained in the `requirements.txt` file:

```bash
conda create --name <env> --file requirements.txt
```

For training the attribute classifier, run the script:

```bash
python attr_classifier.py
```

## 3. Feature Extraction

For extracting discriminative features, run the script:

```bash
python feat_extract_LFWA.py
```

## 4. Construct the dataset

After having the features in the `.npy` format, it is necessary to convert them to the `.pickle` format.

```bash
python LFWA/make_att_file.py
python LFWA/make_feat_file.py
```

## 5. Evaluation

For evaluating purposes, we refer the reader to the [this](https://github.com/CristianoPatricio/zsl-methods) github repository, which contains six state-of-the-art ZSL methods. Having the discriminative features and attributes in the `.pickle` format, you only have to pass it to the ZSL methods.


⚠️ Work in progress...
