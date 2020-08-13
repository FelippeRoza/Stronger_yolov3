# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from xml.etree.ElementTree import parse
import json
from utils.util import ensure_dir
from pathlib import Path

def get_filelists(path, prefix, suffix):
    return glob.glob(os.path.join(path, '{}.{}'.format(prefix, suffix)))

def save_ecp_json(det_list, dest_folder, img_path, id_to_class):
    '''
    det_list: [boxes, labels, scores, var]
    '''
    objects = []
    img_name = Path(img_path).stem
    for index, bbox in enumerate(det_list[0]):
        covs = det_list[3][index]
        obj_class = id_to_class[int(det_list[1][index])]
        if obj_class == 'pedestrian':
            obj_class = 'person'
        obj = {'identity': obj_class,
               'x0': float(bbox[0]),
               'y0': float(bbox[1]),
               'x1': float(bbox[2]),
               'y1': float(bbox[3]),
               'sigma_xmin': float(covs[0]),
               'sigma_ymin': float(covs[1]),
               'sigma_xmax': float(covs[2]),
               'sigma_ymax': float(covs[3]),
               'orient': 0.0,
               'score': float(det_list[2][index])
               }
        objects.append(obj)

    frame = {'identity': 'frame'}
    frame['children'] = objects
    ensure_dir(dest_folder)
    json.dump(frame, open(os.path.join(dest_folder, img_name + '.json'), 'w'), indent=1)


class PascalVocXmlParser(object):
    """Parse annotation for 1-annotation file """

    def __init__(self, annfile, labels):
        self.annfile = annfile
        self.root = self._root_tag(self.annfile)
        self.tree = self._tree(self.annfile)
        self.labels = labels

    def parse(self, filterdiff=True):
        fname = self.get_fname()
        labels, diffcults = self.get_labels()
        boxes = self.get_boxes()
        if filterdiff:
            indices = np.where(np.array(diffcults) == 0)
            boxes = boxes[indices]
            labels=labels[indices]
            return fname, np.array(boxes), labels
        else:
            return fname,boxes,labels,diffcults

    def get_fname(self):
        return os.path.join(self.root.find("filename").text)

    def get_width(self):
        for elem in self.tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self):
        for elem in self.tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self):
        labels = []
        difficult = []
        obj_tags = self.root.findall("object")
        for t in obj_tags:
            labels.append(self.labels.index(t.find("name").text))
            difficult.append(int(t.find("difficult").text))
        return np.array(labels), np.array(difficult)

    def get_boxes(self):
        bbs = []
        obj_tags = self.root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([(float(x1)), (float(y1)), (float(x2)), (float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree
