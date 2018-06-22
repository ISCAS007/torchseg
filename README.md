# torchseg
use pytorch to do image semantic segmentation

# requriments
```
conda install pytorch torchvision -c pytorch
pip install opencv-python
pip install tensorboardX
pip install easydict
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
- [ ] pspnet
- [x] simplenet
- [x] motionnet
- [x] tensorboard
- [x] miou