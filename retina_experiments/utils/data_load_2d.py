import os, os.path as osp
import torch
import numpy as np
import monai.transforms as t
import monai.data as d
import sys
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from skimage import img_as_float32
from sklearn.model_selection import train_test_split

def build_skeleton(im):
    im = im.squeeze().bool()
    if len(im.shape) == 2:
        sk = torch.from_numpy(img_as_float32(skeletonize(im))).unsqueeze(dim=0)
    elif len(im.shape) == 3:
        sk = torch.from_numpy(img_as_float32(skeletonize_3d(im))).unsqueeze(dim=0)
    else:
        sys.exit('bad shape here')

    return sk

def get_transforms_2d_patches(n_classes, n_samples, neg_samples, patch_size):
    label_manager = t.AsDiscreteD(keys=['seg', ], to_onehot=n_classes)
    tr_transforms = t.Compose([
        t.LoadImaged(keys=['img', 'seg'], image_only=True),
        t.EnsureChannelFirstd(keys=['img', 'seg']),
        t.RandCropByPosNegLabeld(keys=('img', 'seg'), label_key='seg', spatial_size=patch_size,
                                 num_samples=n_samples, pos=1, neg=neg_samples), # P(center=fground) = pos/(pos+neg) = 1/(1+neg)
        t.Rotate90d(['img', 'seg']),
        t.RandFlipD(keys=['img', 'seg'], spatial_axis=0, prob=0.2),
        t.RandFlipD(keys=['img', 'seg'], spatial_axis=1, prob=0.2),
        t.ScaleIntensityd(keys=['img', 'seg']),
        t.NormalizeIntensityD(keys=['img']),
        label_manager,
    ])

    vl_transforms = t.Compose([
        t.LoadImaged(keys=['img', 'seg'], image_only=False),
        t.EnsureChannelFirstd(keys=['img', 'seg']),
        t.Rotate90D(['img', 'seg']),
        t.ScaleIntensityd(keys=['img', 'seg']),
        t.NormalizeIntensityD(keys=['img']),
        label_manager,
    ])
    return tr_transforms, vl_transforms

def get_train_val_test_splits(dataset):
    if 'FIVES' in dataset:
        # “n_A/D/G/N.png”, where “n” means the number of images and “A”, “D”, “G”, and “N”
        # stand for “AMD”, “DR”, “Glaucoma” and “Normal”
        im_train_path = osp.join('data/Dataset54_fives/imagesTr/')
        im_list_train = os.listdir(im_train_path)
        im_list_train = sorted([osp.join(im_train_path, n) for n in im_list_train])
        im_test_path = osp.join('data/Dataset54_fives/imagesTs/')
        im_list_test = os.listdir(im_test_path)
        im_list_test = sorted([osp.join(im_test_path, n) for n in im_list_test])

        seg_train_path = osp.join('data/Dataset54_fives/labelsTr')
        seg_list_train = os.listdir(seg_train_path)
        seg_list_train = sorted([osp.join(seg_train_path, n) for n in seg_list_train])
        seg_test_path = osp.join('data/Dataset54_fives/labelsTs')
        seg_list_test = os.listdir(seg_test_path)
        seg_list_test = sorted([osp.join(seg_test_path, n) for n in seg_list_test])

        assert dataset in ['FIVES', 'FIVES_A', 'FIVES_D', 'FIVES_G', 'FIVES_N']

        if dataset != 'FIVES':
            im_list_train = [n for n in im_list_train if n.endswith('_{}_0000.png'.format(dataset[-1]))]
            im_list_test = [n for n in im_list_test if n.endswith('_{}_0000.png'.format(dataset[-1]))]
            seg_list_train = [n for n in seg_list_train if n.endswith('_{}.png'.format(dataset[-1]))]
            seg_list_test = [n for n in seg_list_test if n.endswith('_{}.png'.format(dataset[-1]))]

        im_list_train, im_list_val, seg_list_train, seg_list_val = train_test_split(im_list_train, seg_list_train,
                                                                                    test_size=0.2, random_state=42)

        tr_files = [{"img": img, "seg": seg} for img, seg in zip(im_list_train, seg_list_train)]
        vl_files = [{"img": img, "seg": seg} for img, seg in zip(im_list_val, seg_list_val)]
        test_files = [{"img": img, "seg": seg} for img, seg in zip(im_list_test, seg_list_test)]
        return tr_files, vl_files, test_files
    else:
        sys.exit('bad dataset')

    return tr_files, vl_files, test_files


def get_loaders(dataset, n_samples, neg_samples, n_classes=2, patch_size=(96, 96), cache=0., num_workers=0, tr_percentage=1., ovft_check=0):

    tr_files, vl_files, test_files = get_train_val_test_splits(dataset)

    if tr_percentage < 1.:
        print(40*'-')
        n_tr_examples = len(tr_files)
        random_indexes = np.random.permutation(n_tr_examples)
        kept_indexes = int(n_tr_examples * tr_percentage)
        tr_files = [tr_files[i] for i in random_indexes[:kept_indexes]]
        print('Reducing training data from {} items to {}'.format(n_tr_examples, len(tr_files)))
        print(40 * '-')

    tr_transforms, vl_transforms = get_transforms_2d_patches(n_classes, n_samples, neg_samples, patch_size=patch_size)

    gpu = torch.cuda.is_available()
    if cache>0.:
        # as `RandCropByPosNegLabeld` crops from the cached content and `deepcopy`
        # the crop area instead of modifying the cached value, we can set `copy_cache=False`
        # to avoid unnecessary deepcopy of cached content in `CacheDataset`
        tr_ds = d.CacheDataset(data=tr_files, transform=tr_transforms, cache_rate=cache, num_workers=8, copy_cache=False,)
        vl_ds = d.CacheDataset(data=vl_files, transform=vl_transforms, cache_rate=cache, num_workers=8, copy_cache=False,)
        if ovft_check > 0:
            ovft_ds = d.CacheDataset(data=tr_files[:ovft_check], transform=vl_transforms, cache_rate=cache, copy_cache=False,)
        else:
            ovft_ds = d.CacheDataset(data=tr_files, transform=vl_transforms, cache_rate=cache, copy_cache=False,)

        # disable multi-workers because `ThreadDataLoader` works with multi-threads
        tr_loader = d.ThreadDataLoader(tr_ds, num_workers=0, batch_size=1, shuffle=True)
        vl_loader = d.ThreadDataLoader(vl_ds, num_workers=0, batch_size=1)
        ovft_loader = d.ThreadDataLoader(ovft_ds, num_workers=0, batch_size=1)

    else:
        tr_ds = d.Dataset(data=tr_files, transform=tr_transforms)
        vl_ds = d.Dataset(data=vl_files, transform=vl_transforms)
        tr_loader = d.DataLoader(tr_ds, batch_size=1, num_workers=num_workers, shuffle=True, pin_memory=gpu)
        vl_loader = d.DataLoader(vl_ds, batch_size=1, num_workers=num_workers, pin_memory=gpu)
        if ovft_check > 0: ovft_ds = d.Dataset(data=tr_files[:ovft_check], transform=vl_transforms)
        else: ovft_ds = d.Dataset(data=tr_files, transform=vl_transforms)
        ovft_loader = d.DataLoader(ovft_ds, batch_size=1, num_workers=num_workers, pin_memory=gpu)

    test_ds = d.Dataset(data=test_files, transform=vl_transforms)
    test_loader = d.ThreadDataLoader(test_ds, batch_size=1, num_workers=0, pin_memory=gpu)
    return tr_loader, ovft_loader, vl_loader, test_loader
