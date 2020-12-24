# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from utils import eval_segm
import cv2

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
#        acc_cls = np.diag(hist) / hist.sum(axis=1)
        diag=np.diag(hist)
        acc_cls = np.divide(diag,hist.sum(axis=1),out=np.zeros_like(diag),where=diag!=0)

        acc_cls = np.nanmean(acc_cls)
        iu = np.divide(diag,(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)),out=np.zeros_like(diag),where=diag!=0)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

# may get different result due the use of exact class number
# swap trues and preds should get the same miou
def get_scores(label_trues,label_preds):
    assert label_trues.shape==label_preds.shape,'unmatched trues shape and predicts shape'
    ndim=label_trues.ndim
    shape=label_trues.shape
    height=1
    width=1
    for i in range(ndim):
        if i<ndim-1:
            height*=shape[i]
        else:
            width*=shape[i]

    label_trues_2d=label_trues.reshape((height,width))
    label_preds_2d=label_preds.reshape((height,width))

    cl, n_cl = eval_segm.union_classes(label_preds_2d, label_trues_2d)

    return {'Overall Acc: \t': eval_segm.pixel_accuracy(label_preds_2d,label_trues_2d),
            'Mean Acc : \t': eval_segm.mean_accuracy(label_preds_2d,label_trues_2d),
            'FreqW IoU : \t': eval_segm.frequency_weighted_IU(label_preds_2d,label_trues_2d),
            'Mean IoU : \t': eval_segm.mean_IU(label_preds_2d,label_trues_2d),
            'Appeared cls: \t':n_cl}

def get_fmeasure(gt,pred,fmeasure_only=True):
    if isinstance(gt,str):
        gt_img=cv2.imread(gt,cv2.IMREAD_GRAYSCALE)
    else:
        gt_img=gt

    if isinstance(pred,str):
        pred_img=cv2.imread(pred,cv2.IMREAD_GRAYSCALE)
    else:
        pred_img=pred

    tp=np.sum(np.logical_and(gt_img>0,pred_img>0))
    tn=np.sum(np.logical_and(gt_img==0,pred_img==0))
    fp=np.sum(np.logical_and(gt_img==0,pred_img>0))
    fn=np.sum(np.logical_and(gt_img>0,pred_img==0))

    precision=tp/(tp+fp+1e-5)
    recall=tp/(tp+fn+1e-5)
    fmeasure=2*precision*recall/(precision+recall+1e-5)

    if fmeasure_only:
        return fmeasure
    else:
        return tp,tn,fp,fn,precision,recall,fmeasure