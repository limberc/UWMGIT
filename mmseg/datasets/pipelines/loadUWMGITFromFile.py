from .loading import LoadImageFromFile
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadUWMGITFromFile(LoadImageFromFile):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 max_value=None,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 force_3chan=False):
        super(LoadUWMGITFromFile, self).__init__(to_float32, color_type, file_client_args, imdecode_backend)
        self.max_value = max_value
        self.force_3chan = force_3chan

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            if self.max_value is None:
                img = img.astype(np.float32)
            elif self.max_value == "max":
                img = img.astype(np.float32) / (img.max() + 1e-7)
            else:
                img = img.astype(np.float32) / self.max_value

        if self.force_3chan:
            img = np.stack([img for _ in range(3)], -1)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
