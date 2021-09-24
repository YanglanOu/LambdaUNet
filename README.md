# LambdaUNet

Pytorch Implementation for our MICCAI 2021 paper: [LambdaUNet: 2.5D Stroke Lesion Segmentation of Diffusion-weighted MR Images.](https://arxiv.org/abs/2104.13917)



# Overview
![Loading LambdaUnet Overview](https://github.com/YanglanOu/LambdaUNet/blob/main/images/lambda_layer.png)

# Installation
### Environment
* Tested OS: Linux
* Python >= 3.6

### Dependencies:
1. Install [PyTorch 1.4.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt

    ```

### Datasets
We will release the dataset soon.

# Training
You can train your own models with your customized configs and dataset. For example:

```
python lit_train.py --c samlpe -f 0
```

# Acknowledgment
This repo borrows code from
* [Unet](https://github.com/milesial/Pytorch-UNet)
* [Lambda Networks](https://github.com/lucidrains/lambda-networks)


# Citation
If you find our work useful in your research, please cite our paper:
```
@article{ou2021lambdaunet,
  title={LambdaUNet: 2.5 D Stroke Lesion Segmentation of Diffusion-weighted MR Images},
  author={Ou, Yanglan and Yuan, Ye and Huang, Xiaolei and Wong, Kelvin and Volpi, John and Wang, James Z and Wong, Stephen TC},
  journal={arXiv preprint arXiv:2104.13917},
  year={2021}
}
```

