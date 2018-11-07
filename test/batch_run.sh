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

# for lr in 5e-4 5e-5 2e-4 2e-5
# do
#     python test/pspnet_test.py --batch_size=4 --net_name=pspnet --backbone_freeze=False \
# --midnet_scale=15 --upsample_type=bilinear --backbone_pretrained=True --n_epoch=50 \
# --note=cityscapes_$lr --norm_ways=cityscapes --learning_rate=$lr
# done

for times in 1 2
do
    for midnet_scale in 8 10
    do
        python test/pspnet_test.py --backbone_pretrained=True --backbone_name=vgg19 \
--upsample_type=bilinear --midnet_scale=${midnet_scale} --batch_size=4 --note=midnet_scale024 \
--use_momentum=False --upsample_layer=3 --dataset_name=VOC2012

        python test/pspnet_test.py --backbone_pretrained=True --backbone_name=vgg19 \
--upsample_type=bilinear --midnet_scale=${midnet_scale} --batch_size=4 --note=midnet_scale025 \
--use_momentum=True --upsample_layer=5 --dataset_name=VOC2012
    done
done

# 'dataset.norm_ways':('choices',['caffe','pytorch','cityscapes','-1,1','0,1']),

for times in 1 2 3 4 5
do
    for norm_ways in caffe pytorch cityscapes -1,1 0,1
    do
        python test/pspnet_test.py --batch_size=2 \
        --backbone_pretrained=True --midnet_scale=15 \
        --upsample_type=bilinear --dataset_use_part=320 \
        --note=norm_ways030
    done
done
        
for batch_size in 8 16 32
do
    python test/pspnet_test.py --batch_size=${batch_size} \
    --backbone_pretrained=True --midnet_scale=5 \
    --backbone_freeze=False --backbone_name=vgg16 \
    --upsample_type=bilinear --dataset_use_part=320 \
    --note=bs${batch_size}
done