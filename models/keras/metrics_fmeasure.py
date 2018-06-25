# -*- coding: utf-8 -*-

import keras.backend as K

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    
    y_true with shape of [batch_size,height,width,class_number]
    arg_y_true with shape of [batch_size,height,width] 
    unique(arg_y_true) is subset for range(0,class_number)
    assume 0 is background!
    true positive
    =sum(arg_y_true==arg_y_pred)-sum(arg_y_true==0 and arg_y_pred==0)
    =sum(arg_y_true==arg_y_pred)-sum((arg_y_true+arg_y_pred)==0)

    predict positive
    =sum(arg_y_true>0)
    """
    arg_y_true = K.cast(K.argmax(y_true),K.floatx())
    arg_y_pred = K.cast(K.argmax(y_pred),K.floatx())
    true_positives = K.sum(K.cast(K.equal(arg_y_true,arg_y_pred),K.floatx())) - K.sum(K.cast(K.equal(arg_y_true+arg_y_pred,0),K.floatx()))
    predicted_positives = K.sum(K.cast(K.greater(arg_y_pred,0),K.floatx()))
    
    precision = K.switch(K.equal(predicted_positives,0),
                         K.constant(0.0), # 0 will lower than standard precision, 1 will higher
                         true_positives/predicted_positives)
#    precision = true_positives / (predicted_positives+K.constant(0.1,K.floatx()))
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    arg_y_true = K.cast(K.argmax(y_true),K.floatx())
    arg_y_pred = K.cast(K.argmax(y_pred),K.floatx())
    true_positives = K.sum(K.cast(K.equal(arg_y_true,arg_y_pred),K.floatx())) - K.sum(K.cast(K.equal(arg_y_true+arg_y_pred,0),K.floatx()))
    possible_positives = K.sum(K.cast(K.greater(arg_y_true,0),K.floatx()))
#    recall = true_positives / (possible_positives+K.constant(0.1,K.floatx()))
    recall = K.switch(K.equal(possible_positives,0),
                      K.constant(0.0),
                      true_positives/possible_positives)
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.argmax(y_true)) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r+K.constant(0.1,K.floatx()))
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def k_mean_iou(NUM_CLASSES):
    """
    assume 0 is background for labels and prediction
    labels,prediction with shape of [batch,height,width,class_number]
    
    """
    def switch_mean_iou(labels, predictions):
        mean_iou = K.variable(0.0)
        seen_classes = K.variable(0.0)
    
        for c in range(1,NUM_CLASSES):
            labels_c = K.cast(K.equal(labels, c), K.floatx())
            pred_c = K.cast(K.equal(predictions, c), K.floatx())
    
            labels_c_sum = K.sum(labels_c)
            pred_c_sum = K.sum(pred_c)
    
            intersect = K.sum(labels_c*pred_c)
            union = labels_c_sum + pred_c_sum - intersect
            iou = intersect / union
            condition = K.equal(union, 0)
            mean_iou = K.switch(condition,
                                mean_iou,
                                mean_iou+iou)
            seen_classes = K.switch(condition,
                                    seen_classes,
                                    seen_classes+1)
    
        mean_iou = K.switch(K.equal(seen_classes, 0),
                            mean_iou,
                            mean_iou/seen_classes)
        return mean_iou
    
    return switch_mean_iou

