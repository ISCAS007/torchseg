# torchseg
use pytorch to do image semantic segmentation
- train+val pspnet ```python test/pspnet_test.py --batch_size=8 --net_name=pspnet --augmentation=False --learning_rate=0.01 --optimizer=sgd --backbone_name=resnet101 --midnet_scale=10 --upsample_type=bilinear --backbone_pretrained=True```

# experiments
| net_name | backbone | midnet | suffix | dataset | note | miou(t/v) |
| -------- | -------- | ------ | ------ | ------- | ---- | ---- |
| pspnet | resnet50 | psp | bilinear | cityscapes | ---- | 0.6/0.5 |
| pspent | resnet101 | psp | bilinear | cityscapes | --- | 0.5/? |
| pspnet | - | - | - | - |

作者：cxuan
链接：https://www.jianshu.com/p/7a655e5345b2
來源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

# requriments
- for detail version see [requirements.txt](requirements.txt)
- test on python3
```
conda install pytorch torchvision -c pytorch
pip install opencv-python
pip install tensorboardX
pip install easydict
pip install imgaug
pip install pandas
```

# code reference
- https://github.com/meetshah1995/pytorch-semseg
```
test pspnet model on cityscapes dataset
Overall Acc: 	 0.865857910768811
Mean Acc : 	 0.4482797176755918
FreqW Acc : 	 0.7728509434255326
Mean IoU : 	 0.36876733235206416
note for official website https://hszhao.github.io/projects/pspnet/
the miou is 0.8+
```
- https://github.com/CSAILVision/semantic-segmentation-pytorch
- [cityscapes dataset evaluation](https://github.com/mcordts/cityscapesScripts)

# blog reference
- [paper and dataset for semantic segmentation introduction](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets)

# todo
- [x] [pspnet](models/pspnet.py)
    - [ ] [train on corse dataset and then finetune + optimizer config(not adam)](https://github.com/ZijunDeng/pytorch-semantic-segmentation/issues/6)
    - [ ] [slice/slide + flipped prediction/evaluation](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/issues/12)
    - [x] official pspnet layer setting
        - https://raw.githubusercontent.com/hszhao/PSPNet/master/evaluation/prototxt/pspnet101_cityscapes_713.prototxt
        - [resnet-101](https://dgschwend.github.io/netscope/#/gist/d9f00f2a9703e66c56ae7f2cca970e85) [ethereon](https://ethereon.github.io/netscope/#/gist/d9f00f2a9703e66c56ae7f2cca970e85)
        - [resnet-101-deploy](https://dgschwend.github.io/netscope/#/gist/ace481c81a5faea2a04d5e49dca09150) [ethereon](https://ethereon.github.io/netscope/#/gist/ace481c81a5faea2a04d5e49dca09150)
        - [pspnet101 cityscapes 713](https://dgschwend.github.io/netscope/#/gist/3266b24bf7d2705ae3929b2408774d79) [ethereon](https://ethereon.github.io/netscope/#/gist/3266b24bf7d2705ae3929b2408774d79)
- [x] color for label image and predict image
- [x] [keras empty net: use pytorch loader and tensorboard-pytorch in keras](models/keras/empty_net.py)
- [x] ~[notebook for keras and empty net](notebooks)~
- [x] [simple_model_test](test/simple_model_test.py)
- [x] [motionnet](models/motionnet.py)
- [x] tensorboard-pytorch, [10.0.0.39:6789](10.0.0.39:6789)
- [x] [miou](utils/metrics.py)
- [x] input image preprocess and augmentation
    - [ ] ~~imagenet~~
    - [x] [-1,1]
    - [ ] ~~[0,1]~~
- [x] pspnet + edge
    - [ ] multi input, output, loss, log
    - [ ] edge before pspnet or after pspnet ?
    - [ ] Series connection or Parallel connection
    - [ ] change edge width with epoch
- [x] pspnet + global
    - very different from keras version, from single dilation to multi dilation
- [x] pspnet + dict learning
    - dict net after upsample net (conv feature can be upsampled with duc, but dict feature may not fit for upsample)
- [ ] pspnet + fractal filters (random number of filters)
    - [x] max/mean feature
    - [ ] int parameter a learning for route (choice=int(a), a is the index for right class, for different class, we have different index)
    - [x] before upsample, otherwise will out of memory.
- [x] the ignore_index for CrossEntropyLoss
    - [x] foreground ids[0-18], background ids=ignore_index=255
- [ ] multi outputs metrics support like keras
- [x] [benchmark](test/benchmark_test.py)
    - [x] dataset loader with path
    - [x] upsample predict results
    - [ ] ~~crop + merge predict results~~
    - [x] train id --> label id
    
# benchmark
- use code from https://github.com/mcordts/cityscapesScripts (only support for python2)
- change to python2 environment `source activate env2`
- clone them and install them by `pip install .`
- `export CITYSCAPES_RESULTS=/media/sdc/yzbx/benchmark_output/cityscapes`
- `export CITYSCAPES_DATASET=/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives/gtFine_trainvaltest`
- open terminal and input: `csEvalPixelLevelSemanticLabeling`
- the image size and id transform can be view in [test/benchmark_test.py](test/benchmark_test.py)
- note: the benchmark is very slow, even for only 500 val images, about 5-10s/per image
- a failed result can be see in [#3](https://github.com/ISCAS007/torchseg/issues/3)
