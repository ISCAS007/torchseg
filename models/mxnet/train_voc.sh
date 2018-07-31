#datasets = {
#    'ade20k': ADE20KSegmentation,
#    'pascal_voc': VOCSegmentation,
#    'pascal_aug': VOCAugSegmentation,
#}
python train.py --dataset pascal_voc --model psp --backbone resnet50 --lr 0.001 --checkname resnet50_psp_pascal --ngpus 1