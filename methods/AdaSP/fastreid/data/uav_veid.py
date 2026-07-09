import os.path as osp
import numpy as np
from .bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class UAV_VeID(ImageDataset):
    root = "/home/uit2023/LuuTru/Vutm/dataset/"
    dataset_dir = "UAV-VeID"
    dataset_name = "UAV-VeID"
    
    def __init__(self, root='', verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'images_query_test')
        self.gallery_dir = osp.join(self.dataset_dir, 'images_gallery_test')
        self.split_dir = osp.join(self.dataset_dir, 'UAV-labels')

        self._check_before_run()
        
        train = self._load_data('train_id_label.txt', self.train_dir)
        query = self._load_data('test_query_label.txt', self.query_dir)
        gallery = self._load_data('test_gallery_label.txt', self.gallery_dir)

        super(UAV_VeID, self).__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _load_data(self, split_file, dir_path):
        split_path = osp.join(self.split_dir, split_file)
        if not osp.exists(split_path):
            raise RuntimeError("'{}' is not available".format(split_path))
        
        with open(split_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]

        dataset = []

        for line in lines:
            if len(line) < 2:
                continue
            img_name, pid = osp.basename(line[0]), int(line[1])
            img_path = osp.join(dir_path, img_name)
            
            if pid == -1:
                continue  # Ignore junk images

            dataset.append((img_path, pid, 0))  # camid = 0

        return dataset
