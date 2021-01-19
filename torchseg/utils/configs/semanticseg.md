# semantic segmentation config 


### optimizer and scheduler 
- optimizer: sgd/adam/...

- scheduler: poly/rop/poly_rop(pop)/cos_lr/none
    - poly(poly_lr_scheduler): change learning rate on every step
    - rop(ReduceLROnPlateau)/poly_rop(pop): change learning rate on every epoch 
    according to loss
    - cos_lr(CosineAnnealingLR): change learning rate on every epoch
    - none: not change learning rate

### accumulate gradient to increase "true batch size"
- [how to accumulate gradient](https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient-in-pytorch-i-e-iter-size-in-caffe-prototxt/2522)
- true batch size = accumulate * batch size for cnn layer
- true batch size = batch size for batch norm layer(but we can change momentum for batch norm)
- true epoch number = epoch number / accumulate 
- learning rate can increase for accumulate. 
    - new learning rate * sqrt(batch size) = old learning rate * sqrt(accumulate * batch size)
- better generalization ability with big accumulate