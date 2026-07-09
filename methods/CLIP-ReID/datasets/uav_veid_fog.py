import os.path as osp
import numpy as np
from .bases import BaseImageDataset

class UAV_VeID_Fog(BaseImageDataset):
    dataset_dir = 'UAV-VeID-Fog'

    def __init__(self, root='', verbose=True, **kwargs):
        super(UAV_VeID_Fog, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'images_query_test')
        self.gallery_dir = osp.join(self.dataset_dir, 'images_gallery_test')
        self.split_dir = osp.join(self.dataset_dir, 'UAV-labels')
        
        self.view_file = osp.join(self.split_dir, 'train_view_label(front rear left right).txt')

        self._check_before_run()
        
        self.image_map_view = self._load_view_data()
        
        self.train = self._load_data('train_id_label.txt', self.train_dir, relabel=True)
        self.query = self._load_data('test_query_label.txt', self.query_dir, relabel=False)
        self.gallery = self._load_data('test_gallery_label.txt', self.gallery_dir, relabel=False)

        if verbose:
            print("=> UAV-VeID-Fog dataset loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

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
        if not osp.exists(self.view_file):
            raise RuntimeError("'{}' is not available".format(self.view_file))

    def _load_view_data(self):
        """Load viewpoint annotations"""
        image_map_view = {}
        with open(self.view_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            img_name = parts[0]
            view_label = np.argmax([int(v) for v in parts[1:]])
            image_map_view[img_name] = view_label
        
        return image_map_view

    def _load_data(self, split_file, dir_path, relabel=False):
        split_path = osp.join(self.split_dir, split_file)
        if not osp.exists(split_path):
            raise RuntimeError("'{}' is not available".format(split_path))
        
        with open(split_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]

        dataset = []
        pid_container = set()
        for line in lines:
            if len(line) < 2:
                continue
            img_name, pid = line[0], int(line[1])
            if pid == -1:
                continue  # Ignore junk images
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        for line in lines:
            if len(line) < 2:
                continue
                
            img_name = osp.basename(line[0])
            pid = int(line[1])
            img_path = osp.join(dir_path, img_name)

            if pid == -1:
                continue  # Ignore junk images
            if relabel:
                pid = pid2label[pid]

            viewid = self.image_map_view.get(img_name, 0)
            dataset.append((img_path, pid, 0, viewid))  # camid = 0

        return dataset
