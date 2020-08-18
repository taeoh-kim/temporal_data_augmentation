## Learning Temporally Invariant and Localizable Features via Data Augmentation for Video Recognition (ECCVW 2020)

[Taeoh Kim](https://taeoh-kim.github.io/)<sup>\*1</sup>, [Hyeongmin Lee](https://hyeongminlee.github.io/)<sup>\*1</sup>, [MyeongAh Cho](https://scholar.google.co.kr/citations?user=HLFojbcAAAAJ&hl=ko)<sup>\*1</sup>, [HoSeong Lee](https://hoya012.github.io/)<sup>2</sup>, Dong Heon Cho<sup>2</sup>, Sangyoun Lee<sup>1</sup><br>
<sup>*</sup> indicates equal contribution<br>
<sup>1</sup> Yonsei University<br>
<sup>2</sup> Cognex Deep Learning Lab<br>

## Introduction

Official PyTorch implementation of our paper which has been accepted to 1st Visual Inductive Priors for Data-Efficient Deep Learning Workshop at ECCV 2020 as Oral Presentation.

[Paper Link](https://arxiv.org/abs/2008.05721)

## Environment
- PyTorch 1.4.0
- opencv-python
- PIL (pillow)
- tqdm

## Train
### Prepare training data (Modified UCF-101)

1. Download UCF-101 training data from [UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101.php).

2. Modify training data using [VIPriors Challenge Split](https://github.com/VIPriors/vipriors-challenges-toolkit/tree/master/action-recognition/data).

### How to train

1. Run train.py with following command. (SlowFast-50 Baseline)
    ```bash
    python train.py --dir_data [DB] --out_dir [ExpName] --is_validate
    ```

2. Train on RandAugment-T
    ```bash
    python train.py --dir_data [DB] --out_dir [ExpName] --is_validate --rand_augmentation --aug_mode randaug --randaug_n [N] --randaug_m [M]
    ```

3. Train on Mixing Data Augmentations (cutmix, framecutmix, cubecutmix, mixup, fademixup, cutmixup, framecutmixup, cubecutmixup)
    ```bash
    python train.py --dir_data [DB] --out_dir [ExpName] --is_validate --mix_type [Mix_type]
    ```


### How to test

1. Run train.py with following command.
    ```bash
    python train.py --dir_data [DB] --out_dir [ExpName] --load [ckpt] --test_only --is_validate
    ```

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{kim2020learning,
    title={Learning Temporally Invariant and Localizable Features via Data Augmentation for Video Recognition},
    author={Kim, Taeoh and Lee, Hyeongmin and Cho, MyeongAh and Lee, Ho Seong and Cho, Dong Heon and Lee, Sangyoun},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2020}
}
```
