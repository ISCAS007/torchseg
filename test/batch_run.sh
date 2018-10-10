#!/bin/bash

# for edge_class_num in 2 4 8
# do
#     python test/pspnet_test.py --batch_size=4 --net_name=psp_edge --augmentation=True --learning_rate=0.01 --optimizer=sgd --backbone_name=resnet101 \ # --backbone_freeze=True --midnet_scale=15 --upsample_type=bilinear --backbone_pretrained=True --n_epoch=200 --test=edge \ 
# --edge_class_num=${edge_class_num} --note=edge_class_num${edge_class_num}
# done

# for edge_with_gray in True False
# do
#     python test/pspnet_test.py --batch_size=4 --net_name=psp_edge --augmentation=True --learning_rate=0.01 --optimizer=sgd --backbone_name=resnet50 \
# --backbone_freeze=False --midnet_scale=15 --upsample_type=bilinear --backbone_pretrained=True --n_epoch=200 --test=edge --edge_class_num=2 \
# --edge_with_gray=${edge_with_gray} --note=edge_with_gray_${edge_with_gray}
# done

# for cross_merge_times in 0 1 2
# do
#     python test/pspnet_test.py --test=naive --batch_size=4 --use_reg=True --backbone_pretrained=True \
#     --midnet_scale=15 --upsample_type=bilinear --cross_merge_times=${cross_merge_times} --note=cm${cross_merge_times} 
# done

for lr in 5e-4 5e-5 2e-4 2e-5
do
    python test/pspnet_test.py --batch_size=4 --net_name=pspnet --backbone_freeze=False \
--midnet_scale=15 --upsample_type=bilinear --backbone_pretrained=True --n_epoch=50 \
--note=cityscapes_$lr --norm_ways=cityscapes --learning_rate=$lr
done