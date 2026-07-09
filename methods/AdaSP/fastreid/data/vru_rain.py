import os.path as osp

from .bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VRU_Rain(ImageDataset):
    root = "/home/uit2023/LuuTru/Vutm/dataset/"
    dataset_dir = "VRU-Rain"
    dataset_name = "VRU-Rain"
    
    def __init__(self, root='', verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'images_train')
        self.query_dir = osp.join(self.dataset_dir, 'images_query_big')
        self.gallery_dir = osp.join(self.dataset_dir, 'images_gallery_big')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')

        self._check_before_run()
        
        train = self._load_data('train.txt', self.train_dir, relabel=True)
        query = self._load_data('query_big.txt', self.query_dir, relabel=False)
        gallery = self._load_data('gallery_big.txt', self.gallery_dir, relabel=False)

        super(VRU_Rain, self).__init__(train, query, gallery, **kwargs)
        
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

    def _load_data(self, split_file, dir_path, relabel=False):
        split_path = osp.join(self.split_dir, split_file)
        if not osp.exists(split_path):
            raise RuntimeError("'{}' is not available".format(split_path))

        with open(split_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]

        dataset = []

        for line in lines:
            if len(line) < 2:
                continue
            img_name, pid = line[0], int(line[1])
            img_path = osp.join(dir_path, img_name + ".jpg")

            if pid == -1:
                continue  # Ignore junk images

            dataset.append((img_path, pid, 0))  # Thêm camid = 0

        return dataset
