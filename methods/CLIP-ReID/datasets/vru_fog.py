import os.path as osp

from .bases import BaseImageDataset

class VRU_Fog(BaseImageDataset):
    dataset_dir = 'VRU-Fog'

    def __init__(self, root='', verbose=True, **kwargs):
        super(VRU_Fog, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'images_train')
        self.query_dir = osp.join(self.dataset_dir, 'images_query_small')
        self.gallery_dir = osp.join(self.dataset_dir, 'images_gallery_small')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')

        self._check_before_run()
        
        self.train = self._load_data('train.txt', self.train_dir, relabel=True)
        self.query = self._load_data('query_small.txt', self.query_dir, relabel=False)
        self.gallery = self._load_data('gallery_small.txt', self.gallery_dir, relabel=False)

        if verbose:
            print("=> VRU-Fog dataset loaded")
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
        view_container = set()
        count = 0

        for line in lines:
            if len(line) < 2:
                continue
            img_name, pid = line[0], int(line[1])
            img_path = osp.join(dir_path, img_name + ".jpg")

            if pid == -1:
                continue  # Ignore junk images
            if relabel:
                pid = pid2label[pid]

            viewid = 0
            if hasattr(self, 'image_map_view_train') and img_name in self.image_map_view_train:
                viewid = self.image_map_view_train[img_name]
            elif hasattr(self, 'image_map_view_test') and img_name in self.image_map_view_test:
                viewid = self.image_map_view_test[img_name]
            else:
                count += 1

            view_container.add(viewid)
            dataset.append((img_path, pid, 0, viewid))  # Thêm camid = 0

        print(view_container, 'view_container')
        print(count, 'samples without viewpoint annotations')
        return dataset
