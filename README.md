## Pixel-Adaptive Convolutional Neural Networks

#### [Project page](https://suhangpro.github.io/pac/index.html) |  [Paper](https://arxiv.org/abs/1904.05373) | [Video](https://youtu.be/gsQZbHuR64o)

Pixel-Adaptive Convolutional Neural Networks<br>
[Hang Su](https://suhangpro.github.io/), [Varun Jampani](https://varunjampani.github.io/), [Deqing Sun](http://research.nvidia.com/person/deqing-sun), [Orazio Gallo](https://research.nvidia.com/person/orazio-gallo), [Erik Learned-Miller](http://people.cs.umass.edu/~elm/), and [Jan Kautz](http://jankautz.com/).<br>
CVPR 2019.

### License

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 


### Installation
* Make sure you have Python>=3.5 (we recommend using a Conda environment). 
* Add the project directory to your Python paths.
* Install dependencies:
    * PyTorch v0.4-1.1 (incl. torchvision) with CUDA: see [PyTorch instructions](https://pytorch.org/get-started/locally/).
    * Additional libraries: 
        ```bash
        pip install -r requirements.txt
        ```
* (Optional) Verify installation: 
    ```bash
    python -m unittest 
    ```


### Layer Catalog 

We implemented 5 types of PAC layers (as PyTorch `Module`):  
* `PacConv2d`: the standard variant
* `PacConvTranspose2d`: the transposed (fractionally-strided) variant for upsampling
* `PacPool2d`: the pooling variant
* `PacCRF`: Mean-Field (MF) inference of a CRF
* `PacCRFLoose`: MF inference of a CRF where the MF steps do not share weights

More details regarding each layer is provided below.

#### `PacConv2d`

`PacConv2d` is the PAC counterpart of `nn.Conv2d`. It accepts most standard `nn.Conv2d` arguments (including in_channels, out_channels, kernel_size, bias, stride, padding, dilation, but not groups and padding_mode), 
and we make sure that when the same arguments are used, `PacConv2d` and `nn.Conv2d` have the exact same output sizes. 
A few additional optional arguments are available: 

```
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
```

When used to build computation graphs, this layer takes two input tensors and generates one output tensor:  

```python
in_ch, out_ch, g_ch = 16, 32, 8         # channel sizes of input, output and guidance
f, b, h, w = 5, 2, 64, 64               # filter size, batch size, input height and width
input = torch.rand(b, in_ch, h, w)
guide = torch.rand(b, g_ch, h, w)       # guidance feature ('f' in Eq.3 of paper)

conv = nn.Conv2d(in_ch, out_ch, f)
out_conv = conv(input)                  # standard spatial convolution

pacconv = PacConv2d(in_ch, out_ch, f)   
out_pac = pacconv(input, guide)         # PAC 
out_pac = pacconv(input, None, guide_k) # alternative interface
                                        # guide_k is pre-computed 'K' (see Eq.3 of paper) 
                                        # of shape [b, g_ch, f, f, h, w]. packernel2d can be 
                                        # used for its creation.  
```

Use `pacconv2d` (in conjunction with `packernel2d`) for its functional interface. 

#### `PacConvTranspose2d`
`PacConvTranspose2d` is the PAC counterpart of `nn.ConvTranspose2d`. It accepts most standard `nn.ConvTranspose2d` 
arguments (including in_channels, out_channels, kernel_size, bias, stride, padding, output_padding, dilation, but not groups and padding_mode), and we make sure that when the same arguments are used, 
`PacConvTranspose2d` and `nn.ConvTranspose2d` have the exact same output sizes. 
A few additional optional arguments are available: , and also a few additional ones: 

```
    Args (in addition to those of ConvTranspose2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform' | 'linear'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
```

Similar to `PacConv2d`, `PacConvTranspose2d` also offers two ways of usage: 

```python
in_ch, out_ch, g_ch = 16, 32, 8             # channel sizes of input, output and guidance
f, b, h, w, oh, ow = 5, 2, 8, 8, 16, 16     # filter size, batch size, input height and width
input = torch.rand(b, in_ch, h, w)
guide = torch.rand(b, g_ch, oh, ow)         # guidance feature, note that it needs to match 
                                            # the spatial sizes of the output

convt = nn.ConvTranspose2d(in_ch, out_ch, f, stride=2, padding=2, output_padding=1)
out_convt = convt(input)                    # standard transposed convolution

pacconvt = PacConvTranspose2d(in_ch, out_ch, f, stride=2, padding=2, output_padding=1)   
out_pact = pacconvt(input, guide)           # PAC 
out_pact = pacconvt(input, None, guide_k)   # alternative interface
                                            # guide_k is pre-computed 'K' 
                                            # of shape [b, g_ch, f, f, oh, ow].
                                            # packernel2d can be used for its creation.  
```

Use `pacconv_transpose2d` (in conjunction with `packernel2d`) for its functional interface. 

#### `PacPool2d`
`PacPool2d` is the PAC counterpart of `nn.AvgPool2d`. It accepts most standard `nn.AvgPool2d` 
arguments (including kernel_size, stride, padding, dilation, but not ceil_mode and count_include_pad), and we make sure that when the same arguments are used, 
`PacPool2d` and `nn.AvgPool2d` have the exact same output sizes. 
A few additional optional arguments are available: , and also a few additional ones: 

```
    Args:
        kernel_size, stride, padding, dilation
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        channel_wise (bool): Default: False
        normalize_kernel (bool): Default: False
        out_channels (int): needs to be specified for channel_wise 'inv_*' (non-fixed) kernels. Default: -1

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
```

Similar to `PacConv2d`, `PacPool2d` also offers two ways of usage: 

```python
in_ch, g_ch = 16, 8                     # channel sizes of input and guidance
stride, f, b, h, w = 5, 2, 64, 64       # stride, filter size, batch size, input height and width
input = torch.rand(b, in_ch, h, w)
guide = torch.rand(b, g_ch, h, w)       # guidance feature 

pool = nn.AvgPool2d(f, stride)
out_pool = pool(input)                  # standard spatial convolution

pacpool = PacPool2d(f, stride)   
out_pac = pacpool(input, guide)         # PAC 
out_pac = pacpool(input, None, guide_k) # alternative interface
                                        # guide_k is pre-computed 'K'
                                        # of shape [b, g_ch, f, f, h, w]. packernel2d can be 
                                        # used for its creation.  
```

Use `pacpool2d` (in conjunction with `packernel2d`) for its functional interface. 

#### `PacCRF` and `PacCRFLoose`
These layers offer a convenient way to add a CRF component at the end of a dense prediction network. 
They performs approximate mean-field inference under the hood. Available arguments 
include: 

```python
    Args:
        channels (int): number of categories.
        num_steps (int): number of mean-field update steps.
        final_output (str): 'log_softmax' | 'softmax' | 'log_Q'. Default: 'log_Q'
        perturbed_init (bool): whether to perturb initialization. Default: True
        native_impl (bool): Default: False
        fixed_weighting (bool): whether to use fixed weighting for unary/pairwise terms. Default: False
        unary_weight (float): Default: 1.0
        pairwise_kernels (dict or list): pairwise kernels, see add_pairwise_kernel() for details. Default: None
```

Usage example: 

```python
# create a CRF layer for 21 classes using 5 mean-field steps
crf = PacCRF(21, num_steps=5, unary_weight=1.0)

# add a pariwise term with equal weight with the unary term
crf.add_pairwise_kernel(kernel_size=5, dilation=1, blur=1, compat_type='4d', pairwise_weight=1.0)

# a convenient function is provided for creating pairwise features based on pixel color and positions
edge_features = [paccrf.create_YXRGB(im, yx_scale=100.0, rgb_scale=30.0)] 
output = crf(unary, edge_features)

# Note that we use constant values for unary_weight, pairwise_weight, yx_scale, rgb_scale, but they can 
# also take tensors and be learned through backprop.
```

### Experiments

#### Joint upsampling
##### Joint depth upsampling on NYU Depth V2
* Train/test split is provided by [Li et al.](https://sites.google.com/site/yijunlimaverick/deepjointfilter) 
* Test with one of our pre-trained models: 

    ```bash
    python -m task_jointUpsampling.main --load-weights weights_depth/x8_pac_weights_epoch_5000.pth \
                                        --download \
                                        --factor 8 \
                                        --model PacJointUpsample \
                                        --dataset NYUDepthV2 \
                                        --data-root data/nyu
    ```
    
    |   | 4x | 8x  | 16x  |
    |---|---|---|---|
    | `Bilinear` | RMSE: 5.43 | RMSE: 8.36  | RMSE: 12.90 |
    | `PacJointUpsample`  | RMSE: 2.39 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x4_pac_weights_epoch_5000.pth) | RMSE: 4.59 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x8_pac_weights_epoch_5000.pth)  | RMSE: 8.09 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x16_pac_weights_epoch_5000.pth)  |
    | `PacJointUpsampleLite`  | RMSE: 2.55  &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x4_paclite_weights_epoch_5000.pth)  | RMSE: 4.82  &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x8_paclite_weights_epoch_5000.pth)  | RMSE: 8.52 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x16_paclite_weights_epoch_5000.pth)  |
    | `DJIF`  | RMSE: 2.64 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x4_djif_weights_epoch_5000.pth)  | RMSE: 5.15 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x8_djif_weights_epoch_5000.pth)  | RMSE: 9.39 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/depth_upsampling/x16_djif_weights_epoch_5000.pth)  |

* Train from scratch: 

    ```bash
    python -m task_jointUpsampling.main --factor 8 \
                                        --data-root data/nyu \
                                        --exp-root exp/nyu \
                                        --download \
                                        --dataset NYUDepthV2 \
                                        --epochs 5000 \
                                        --lr-steps 3500 4500
    ```
    
    See `python -m task_jointUpsampling.main -h` for the complete list of command line options. 

##### Joint optical flow upsampling on Sintel
* Train/val split (`1` - train, `2` - val) is provided in [meta/Sintel_train_val.txt](task_jointUpsampling/meta/Sintel_train_val.txt) ([original source](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/Sintel_train_val.txt)): 
    * Validation: 133 pairs
        * `ambush_6` (all 19)
        * `bamboo_2` (last 25)
        * `cave_4` (last 25)
        * `market_6` (all 39)
        * `temple_2` (last 25)
    * Training: remaining 908 pairs

* Test with one of our pre-trained models: 

    ```bash
    python -m task_jointUpsampling.main --load-weights weights_flow/x8_pac_weights_epoch_5000.pth \
                                        --download \
                                        --factor 8 \
                                        --model PacJointUpsample \
                                        --dataset Sintel \
                                        --data-root data/sintel
    ```
    
    |   | 4x | 8x  | 16x  |
    |---|---|---|---|
    | `Bilinear` | EPE: 0.4650 | EPE: 0.9011 | EPE: 1.6281 |
    | `PacJointUpsample` | EPE: 0.1042 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/flow_upsampling/x4_pac_weights_epoch_5000.pth) | EPE: 0.2558 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/flow_upsampling/x8_pac_weights_epoch_5000.pth) | EPE: 0.5921 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/flow_upsampling/x16_pac_weights_epoch_5000.pth) |
    | `DJIF` | EPE: 0.1760 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/flow_upsampling/x4_djif_weights_epoch_5000.pth) | EPE: 0.4382 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/flow_upsampling/x8_djif_weights_epoch_5000.pth) | EPE: 1.0422 &#124; [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/flow_upsampling/x16_djif_weights_epoch_5000.pth) |

* Train from scratch: 

    ```bash
    python -m task_jointUpsampling.main --factor 8 \
                                        --data-root data/sintel \
                                        --exp-root exp/sintel \
                                        --download \
                                        --dataset Sintel \
                                        --epochs 5000 \
                                        --lr-steps 3500 4500
    ```
    
    See `python -m task_jointUpsampling.main -h` for the complete list of command line options. 
    
#### Semantic segmentation
* Test with one of the pre-trained models:

    ```bash
    python -m task_semanticSegmentation.main --data-root data/voc \ 
                                             --exp-root exp/voc \
                                             --download \
                                             --load-weights fcn8s_from_caffe.pth \
                                             --model fcn8s \
                                             --test-split val11_sbd \
                                             --test-crop -1
    ```
    
    |   | miou (val / test) | model name  | weights | 
    |---|---|---|---|
    | Backbone (FCN8s)  | 65.51% / 67.20% | `fcn8s`  | [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/semantic_segmentation/fcn8s_from_caffe.pth)   |
    | PacCRF  | 68.90% / 69.82% | `fcn8s_crfi5p4d5641p4d5161`  | [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/semantic_segmentation/fcn8s_paccrf_epoch_30.pth)  |
    | PacCRF-32  |  68.52% / 69.41% | `fcn8s_crfi5p4d5321`  | [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/semantic_segmentation/fcn8s_paccrf32_epoch_30.pth)  |
    | PacFCN (hot-swapping)  | 67.44% / 69.18% | `fcn8spac`  | [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/semantic_segmentation/fcn8s_pacfcn_epoch_20.pth)  |
    | PacFCN+PacCRF  |  69.87% / 71.34% | `fcn8spac_crfi5p4d5641p4d5161`  |  [download](http://maxwell.cs.umass.edu/pacnet-data/pretrained_weights/semantic_segmentation/fcn8s_pacfcncrf_epoch_20.pth) |
    
    Note that the last two models requires argument `--test-crop 512`.

* Generate predictions
    
    Use the `--eval pred` mode to save predictions instead of reporting scores. Predictions will be saved 
    under `exp-root`/outputs_*_pred, and can be used for VOC evaluation server: 
    
    ```bash
    python -m task_semanticSegmentation.main \
    --data-root data/voc \
    --exp-root exp/voc \
    --load-weights fcn8s_paccrf_epoch_30.pth \
    --test-crop -1 \
    --test-split test \
    --eval pred \
    --model fcn8s_crfi5p4d5641p4d5161 
    
    cd exp/voc
    mkdir -p results/VOC2012/Segmentation
    mv outputs_test_pred results/VOC2012/Segmentation/comp6_test_cls
    tar zcf results_fcn8s_crf.tgz results
    ```
    
    Note that since there is no publicly available URL for the test split of VOC, when using the test split, 
    the data files need to be downloaded from the [official website](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar) manually. 
    Simply place the downloaded VOC2012test.tar under the data root and untar. 

* Train models

    As an example, here shows the commands for the two-stage training of PacCRF: 
    
    ```bash
    # stage 1: train CRF only with frozen backbone
    python -m task_semanticSegmentation.main \
    --data-root data/voc \
    --exp-root exp/voc/crf_only \
    --load-weights-backbone fcn8s_from_caffe.pth \
    --train-split train11 \
    --test-split val11_sbd \
    --train-crop 449 \
    --test-crop -1 \
    --model fcn8sfrozen_crfi5p4d5641p4d5161 \
    --epochs 40 \
    --lr 0.001 \
    --lr-steps 20
    
    # stage 2: train CRF and backbone jointly
    python -m task_semanticSegmentation.main \
    --data-root data/voc \
    --exp-root exp/voc/joint \
    --load-weights-backbone fcn8s_from_caffe.pth \
    --load-weights exp/voc/crf_only/weights_epoch_40.pth \
    --train-split train11 \
    --test-split val11_sbd \
    --train-crop 449 \
    --test-crop -1 \
    --model fcn8s_crfi5lp4d5641p4d5161 \
    --epochs 30 \
    --lr 0.0000001 \
    --lr-steps 20
    ```

See `python -m task_semanticSegmentation.main -h` for the complete list of command line options. 

### Citation
If you use this code for your research, please consider citing our paper: 
```
@inproceedings{su2019pixel,
  author    = {Hang Su and 
	       Varun Jampani and 
	       Deqing Sun and 
	       Orazio Gallo and 
	       Erik Learned-Miller and 
	       Jan Kautz},
  title     = {Pixel-Adaptive Convolutional Neural Networks},
  booktitle = {Proceedings of the IEEE Conference on Computer 
               Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```
