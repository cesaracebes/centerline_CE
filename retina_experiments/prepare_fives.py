
# similar to https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py

import argparse, os, os.path as osp
from shutil import copyfile
from tqdm import tqdm
from skimage import io

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--nnUNet_raw', type=str, help='nnUNet_raw path')
parser.add_argument('--source', type=str, help='fives')
parser.add_argument('--dataset_name', type=str, help='Dataset54_fives')

def build_dataset(nnUNet_raw, source, dataset_name):
    imagestr = osp.join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = osp.join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = osp.join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = osp.join(nnUNet_raw, dataset_name, 'labelsTs')
    os.makedirs(imagestr, exist_ok=True)
    os.makedirs(imagests, exist_ok=True)
    os.makedirs(labelstr, exist_ok=True)
    os.makedirs(labelsts, exist_ok=True)

    path_tr_ims = osp.join(source, 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation', 'train', 'Original')
    path_tr_segs = osp.join(source, 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation', 'train', 'Ground truth')

    tr_im_list = os.listdir(path_tr_ims)
    tr_im_list = [osp.join(path_tr_ims, n) for n in tr_im_list if '.png' in n]
    tr_seg_list = [n.replace(path_tr_ims, path_tr_segs) for n in tr_im_list]

    path_test_ims = osp.join(source, 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation', 'test', 'Original')
    path_test_segs = osp.join(source, 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation', 'test', 'Ground truth')

    test_im_list = os.listdir(path_test_ims)
    test_im_list = [osp.join(path_test_ims, n) for n in test_im_list]

    test_seg_list = [n.replace(path_test_ims, path_test_segs) for n in test_im_list]

    for i in tqdm(range(len(tr_im_list))):
        n = tr_im_list[i]
        n_out = n.replace(path_tr_ims, imagestr).replace('.png', '_0000.png')
        copyfile(n, n_out)
    for i in tqdm(range(len(tr_seg_list))):
        s = tr_seg_list[i]
        s_out = s.replace(path_tr_segs, labelstr)
        x = io.imread(s)[:, :, 0]
        x[x == 255] = 1
        io.imsave(s_out, x, check_contrast=False)

    for i in tqdm(range(len(test_im_list))):
        n = test_im_list[i]
        n_out = n.replace(path_test_ims, imagests).replace('.png', '_0000.png')
        copyfile(n, n_out)

    for i in tqdm(range(len(test_seg_list))):
        n = test_seg_list[i]
        n_out = n.replace(path_test_segs, labelsts)
        x = io.imread(n)[:, :, 0]
        x[x == 255] = 1
        io.imsave(n_out, x, check_contrast=False)



if __name__ == '__main__':

    args = parser.parse_args()
    build_dataset(args.nnUNet_raw, args.source, args.dataset_name)


