import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)



import h5py
import torch
from torch.utils import data
import numpy as np
from pycocotools import mask as m
from pathlib import Path
import cv2
import ast
import os
@DATASETS.register_module
class Hdf5Dataset(Dataset):
    CLASSES = None
    def __init__(self, ann_file, pipeline, data_root=None, img_prefix='', test_mode=False,):
        super().__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.baseDir = Path(img_prefix)
        self.chunkSize = 2000 
        assert self.baseDir.is_dir()
        self.files = sorted(self.baseDir.glob('*.h5'), key=lambda x: int(x.parts[-1][-6:-3]))
        self.img_infos={'witdh':512, 'height':512}
        with open(ann_file, 'r') as f:
            self.totalData = int(f.read())
        self.data_info=[]
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self._set_group_flag()


    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 0

    # def pre_pipeline(self, results):
    #     results['seg_fields'] =    []
    #     results['img'] = img
    #     results['img_shape'] = img.shape
    #     results['ori_shape'] = img.shape
    #     #results['gt_bboxes'] = ann_info['bboxes']
    #     #results['bbox_fields'].append('gt_bboxes')
    #     results['gt_semantic_seg'] = mask
    #     r#esults['seg_fields'].append('gt_semantic_seg')
    #     #results['img_prefix'] = self.img_prefix
    #     #results['seg_prefix'] = self.seg_prefix
    #     #results['proposal_file'] = self.proposal_file
    #     #results['bbox_fields'] = []
    #     #results['mask_fields'] = []
    #     #results['seg_fields'] = []

    # - "img_shape": shape of the image input to the network as a tuple
    #     (h, w, c).  Note that images may be zero padded on the bottom/right
    #     if the batch tensor is larger than this shape.

    # - "scale_factor": a float indicating the preprocessing scale

    # - "flip": a boolean indicating if image flip transform was used

    # - "filename": path to the image file

    # - "ori_shape": original shape of the image as a tuple (h, w, c)

    # - "pad_shape": image shape after padding

    # - "img_norm_cfg": a dict of normalization information:
    #     - mean - per channel mean subtraction
    #     - std - per channel std divisor
    #     - to_rgb - bool indicating if bgr was converted to rgb

    def __len__(self):
        
        return self.totalData


    def __getitem__(self, idx):
        """
        dataset.attrs['polygon1'] is hair mask
        dataset.attrs['polygon2'] is face mask
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            #self.overlayMask(data)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        fidx = idx//self.chunkSize
        with h5py.File(self.files[fidx]) as f:
            dataset = f['group{0:03}'.format(fidx)]['{}'.format(idx)]
            img = cv2.imdecode(np.array(dataset), flags=cv2.IMREAD_COLOR)
            results ={
                'img': img,
                'img_shape': img.shape,
                'ori_shape': img.shape,
                'filename': os.path.join(self.files[fidx],'group{0:03}'.format(fidx),'{}'.format(idx))
                # 'gt_bboxes' ,
                #  'gt_labels'
                
            }
            for k in ['polygon1', 'polygon2']:
                if k in dataset.attrs.keys():
                    results['gt_masks']=m.decode(ast.literal_eval(dataset.attrs[k]))
            self.overlayMask(results)
                    
        return self.pipeline(results)

    # def prepare_test_img(self, idx):
    #     img_info = self.img_infos[idx]
    #     results = dict(img_info=img_info)
    #     if self.proposals is not None:
    #         results['proposals'] = self.proposals[idx]
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)


    # def __getitem__(self, item):
    #     """
    #     dataset.attrs['polygon1'] is hair mask
    #     dataset.attrs['polygon2'] is face mask
    #     """
    #     fidx = item//self.chunkSize
    #     with h5py.File(self.files[fidx]) as f:
    #         dataset = f['group{0:03}'.format(fidx)]['{}'.format(item)]
    #         data ={
    #             'image':  cv2.imdecode(np.array(dataset), flags=cv2.IMREAD_COLOR)
    #         }
    #         for k in ['polygon1', 'polygon2']:
    #             if k in dataset.attrs.keys():
    #                 data[k]=m.decode(ast.literal_eval(dataset.attrs[k]))

    #         # attributes string values
    #         #for k, val in dataset.attrs.items():
    #         #    data[k] = val

    #         #display mask overlay image
    #         #self.overlayMask(data)
    #     return data

    def overlayMask(self, data):
        img, hair_mask  = data['img'], data['gt_masks']
        img[:,:,1][hair_mask!=0]=255
        #cv2.imwrite('example.jpg',img)
        cv2.imwrite('img.jpg',img)
        # cv2.waitKey(0)
        


if __name__ == "__main__":
    root = '/data/hairData' #this path needs to be modified
    dataset = Hdf5Dataset(root)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    for batch in loader:
        print(batch['image'].shape)