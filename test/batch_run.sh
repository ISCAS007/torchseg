#!/bin/bash

# for edge_class_num in 2 4 8
# do
#     python test/pspnet_test.py --batch_size=4 --net_name=psp_edge --augmentation=True --learning_rate=0.01 --optimizer=sgd --backbone_name=resnet101 \ # --backbone_freeze=True --midnet_scale=15 --upsample_type=bilinear --backbone_pretrained=True --n_epoch=200 --test=edge \ 
# --edge_class_num=${edge_class_num} --note=edge_class_num${edge_class_num}
# done

pwd
echo $CUDA_VISIBLE_DEVICES
for edge_seg_order in later first
do
    python test/pspnet_test.py --batch_size=4 --net_name=psp_edge --augmentation=True --learning_rate=0.01 --optimizer=sgd --backbone_name=resnet50 \
--backbone_freeze=True --midnet_scale=15 --upsample_type=bilinear --backbone_pretrained=True --n_epoch=200 --test=edge --edge_class_num=2 \
--edge_seg_order=${edge_seg_order} --note=edge512_${edge_seg_order}
done