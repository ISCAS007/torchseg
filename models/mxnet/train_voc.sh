#datasets = {
#    'ade20k': ADE20KSegmentation,
#    'pascal_voc': VOCSegmentation,
#    'pascal_aug': VOCAugSegmentation,
#}
# batch-size=4 ==> use-gpu-memory = 5311M
# Epoch 49, validation pixAcc: 0.789, mIoU: 0.519: 100%|█| 91/91 [01:44<00:00,  1.15s/it]
python train.py --dataset pascal_voc --model psp --backbone resnet50 --lr 0.001 --checkname resnet50_psp_pascal --batch-size 4