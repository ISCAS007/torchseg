# torchseg
### tag v0.4 use pytorch0.4
use pytorch to do image semantic segmentation
- train+val pspnet ```python test/pspnet_test.py --batch_size=8 --net_name=pspnet --augmentation=False --learning_rate=0.01 --optimizer=sgd --backbone_name=resnet101 --midnet_scale=10 --upsample_type=bilinear --backbone_pretrained=True```

### current version use pytorch1.1+ (for DDP and sync batchnorm support)
- DDP trainning
```
python test/pspnet_test.py --test dist --mp_dist True --note dist --batch_size 4
```
- support sync batchnorm
```
python test/pspnet_test.py --test dist --mp_dist True --note dist --batch_size 4 --use_sync_bn True
```

# experiments
| net_name    | backbone  | midnet | suffix   | dataset    | note | miou(t/v) |
| --------    | --------- | ------ | -------- | ---------- | ---- | --------- |
| pspnet      | resnet50  | psp    | bilinear | cityscapes |  1   | 0.6/0.5   |
| pspent      | resnet101 | psp    | bilinear | cityscapes |  1   | 0.75/0.47 |
| pspent      | resnet101 | psp    | bilinear | cityscapes |  3   | 0.80/0.50 |
| psp_convert | resnet101 | psp    | bilinear | cityscapes |  4   | -/- |
| psp_edge    | resnet101 | psp    | bilinear | cityscapes |  3   | -/- |

1. 100 epoch
3. 200 epoch
4. require GPU number >=2, input_shape=(713,713)

# [offical result](https://hszhao.github.io/projects/pspnet/)
| net_name | backbone | dataset | note | miou |
| -------- | -------- | ------- | ---- | ---- |
| pspnet   |resnet101 | voc     | -    | 82.6 |
| pspnet   |resnet101 | voc     | coco | 85.4 |
| pspnet   |resnet101 |cityscape| -    | 78.4 |
| pspnet   |resnet101 |cityscape|coarse| 80.2 |
| pspnet   |resnet50  |ade20k   | ms   | 42.78|
|deeplabv3+|xception  | voc     | -    | 87.8 |
|deeplabv3+|xception  | voc     | JFT  | 89.0 |

# requriments
- for detail version see [requirements.txt](requirements.txt)
- test on python3
```
conda install pytorch=1.1 torchvision cudatoolkit=9.2 -c pytorch
pip install opencv-python
pip install tensorboardX
pip install easydict
pip install imgaug
pip install pandas
pip install torchsummary
...
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
- [dual attention](https://github.com/junfu1115/DANet)

# blog reference
- [paper and dataset for semantic segmentation introduction](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets)

# benchmark

| dataset | train | val | test | class | resolution |
| - | - | - | - | - | - |
| cityscapes | 2975 | 500 | 1525 | 19 | 1024x2048 |
| pascal-context | 4998 | - | 5105 | 59 | - |
| pascal-voc2012 | 1464 | 1449 | 1456 | 20 | - |
| camvid | 369 | 101 | 233 | 11 | - |

## cityscapes
- use code from https://github.com/mcordts/cityscapesScripts (only support for python2)
- change to python2 environment `source activate env2`
- clone them and install them by `pip install .`
- `export CITYSCAPES_RESULTS=/media/sdc/yzbx/benchmark_output/cityscapes`
- `export CITYSCAPES_DATASET=/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives/gtFine_trainvaltest`
- open terminal and input: `csEvalPixelLevelSemanticLabeling`
- the image size and id transform can be view in [test/benchmark_test.py](test/benchmark_test.py)
- note: the benchmark is very slow, even for only 500 val images, about 5-10s/per image
- a failed result can be see in [#3](https://github.com/ISCAS007/torchseg/issues/3)

## voc2012
```
rm -rf output/results/VOC2012/Segmentation/comp6_test_cls
mv xxx output/results/VOC2012/Segmentation/comp6_test_cls
tar -czvf results.tgz results
```


# useful script
- `python utils/summary_to_csv.py` output best val/iou to orange tab file
- `python test/dp_log.py` view pid and detail
- `pipreqs --ignore models/examples,models/mxnet,utils/model_hyperopt.py --force .` pip requirements
- `python test/fbms_train.py` motion segmentation
- `python test/pspnet_test.py` semantic segmetnation
- `python notbooks/motion_statistic.py --app dump_tasks --note xxx --dump_group` dump motion segmentation experiment in table format.