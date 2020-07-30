import os, sys, glob
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "."))

import numpy as np
from utils.dataset_util import PascalVocXmlParser
import cv2
from dataset.augment import transform
import random
import torch
from torch.utils.data import DataLoader
import os.path as osp
import dataset.augment.dataAug  as dataAug
import xml.etree.ElementTree as ET
from dataset.BaseDataset import BaseDataset


class KITTIdataset(BaseDataset):
    def __init__(self,cfg,subset, istrain):
        super().__init__(cfg,subset,istrain)
        self._annopath = os.path.join('{}', 'labels', '{}.xml')
        self._imgpath = os.path.join('{}', 'images', '{}.png')
        self.istrain = istrain
        self._ids = []
        rootpath = os.path.join(self.dataset_root, subset)
        self.labels = ['Tram', 'Truck', 'Cyclist', 'DontCare', 'person', 'car', 'Misc', 'Van', 'Person_sitting']

        for f in glob.glob(os.path.join(rootpath, 'labels', "*.xml")):
            base=os.path.basename(f)
            self._ids.append((rootpath, os.path.splitext(base)[0]))

    def __len__(self):
        return len(self._ids) // self.batch_size

    def _parse_annotation(self,itemidx,random_trainsize):
        rootpath, filename = self._ids[itemidx]
        annpath = self._annopath.format(rootpath, filename)
        imgpath = self._imgpath.format(rootpath, filename)
        fname, bboxes, labels = PascalVocXmlParser(annpath, self.labels).parse()
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if self.istrain:
            img, bboxes = dataAug.random_horizontal_flip(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.random_crop(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.random_translate(np.copy(img), np.copy(bboxes))
        ori_shape=img.shape[:2]
        img, bboxes = dataAug.img_preprocess2(np.copy(img), np.copy(bboxes),
                                              (random_trainsize, random_trainsize), True)
        return img,bboxes,labels,imgpath,ori_shape

def get_dataset(cfg):
    trainset = KITTIdataset(cfg, 'train', istrain=True)
    trainset = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=cfg.DATASET.numworker, pin_memory=True)

    valset = KITTIdataset(cfg, 'val', istrain=False)
    valset = DataLoader(dataset=valset, batch_size=1, shuffle=False, num_workers=cfg.DATASET.numworker, pin_memory=True)
    return trainset, valset


if __name__ == '__main__':
    from yacscfg import _C as cfg
    import os
    import argparse
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        default='configs/strongerv3_kl.yaml'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.EVAL.iou_thres = 0.5
    cfg.DATASET.dataset_root='/data/datasets/kitti/kitti_split'
    cfg.freeze()
    train, val = get_dataset(cfg)
    print('Train set size:', len(train))
    print('Val set size:', len(val))
