"""
the label for huawei segmentation dataset, the same as the categories for cityscapes

"""
huawei_names='void,flat,human,vehicle,construction,object,nature,sky'
huawei_labels=huawei_names.split(',')
    
def label2id(label):
    
    return huawei_labels.index(label)
