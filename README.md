# Zero-shot face recognition: Improving the discriminability of visual face features using a Semantic-Guided Attention Model

This is the official implementation of the paper [Zero-shot face recognition: Improving the discriminability of visual face features using a Semantic-Guided Attention Model](https://www.sciencedirect.com/science/article/pii/S0957417422016803).

<img src="https://github.com/CristianoPatricio/SGAM/blob/main/figures/sgam_model.jpg" width="600" style="text-align:center;">

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

For evaluating purposes, we refer the reader to the [this](https://github.com/CristianoPatricio/zsl-methods) github repository, which contains six state-of-the-art ZSL methods. Having the discriminative features in the `.pickle` format, you only have to pass it to the ZSL methods.

⚠️ Work in progress...
