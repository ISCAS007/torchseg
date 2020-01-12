import pydensecrf.densecrf as dcrf
import numpy as np
import fire
import os

from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import glob
from tqdm import tqdm

def apply_crf(image_dir,result_dir,output_dir=None):
    if output_dir is None:
        output_dir=result_dir+'_crf'

    result_masks=glob.glob(os.path.join(result_dir,'**','*.png'),recursive=True)
    assert len(result_masks)>0

    img_suffix='.jpg'
    mask_suffix='.png'
    tqdm_set=tqdm(result_masks)
    for result_file in tqdm_set:
        img_file=result_file.replace(result_dir,image_dir).replace(mask_suffix,img_suffix)

        assert os.path.exists(img_file),f'{img_file}'
        img=imread(img_file)

        anno_rgb=imread(result_file).astype(np.uint32)

        min_val = np.min(anno_rgb.ravel())
        max_val = np.max(anno_rgb.ravel())

        if max_val==min_val:
            output_file=result_file.replace(result_dir,output_dir)
            os.makedirs(os.path.dirname(output_file),exist_ok=True)
            imsave(output_file, anno_rgb.astype(np.uint8))
            output_file_info=output_file.split(os.sep)
            tqdm_set.set_postfix(video=output_file_info[-2],file=output_file_info[-1])
            continue

        out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)
        labels = np.zeros((2, img.shape[0], img.shape[1]))
        labels[1, :, :] = out
        labels[0, :, :] = 1 - out

        # colors = [0, 255]
        colors=[0,1]
        colorize = np.empty((len(colors), 1), np.uint8)
        colorize[:,0] = colors

        n_labels = 2

        crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        U = unary_from_softmax(labels)
        crf.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])

        crf.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats = create_pairwise_bilateral(sdims=(60, 60), schan=(5, 5, 5),
                                      img=img, chdim=2)
        crf.addPairwiseEnergy(feats, compat=5,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = crf.inference(5)

        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP]

        output_file=result_file.replace(result_dir,output_dir)
        os.makedirs(os.path.dirname(output_file),exist_ok=True)

        imsave(output_file, MAP.reshape(anno_rgb.shape))
#        print(f"Saving to: {output_file}" )
        output_file_info=output_file.split(os.sep)
        tqdm_set.set_postfix(video=output_file_info[-2],file=output_file_info[-1])

if __name__ == '__main__':
    """
    # for FBMS

    # for DAVIS2017
    python utils/applyCRF.py apply_crf ~/cvdataset/DAVIS/JPEGImages/480p ~/tmp/result/DAVIS2017/val/test_inn_attd
    """
    fire.Fire()