import pdb

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
import os
from collections import OrderedDict
import torch
import numpy as np
import cv2
from prettytable import PrettyTable
from PIL import Image
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import os.path as osp
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics


@DATASETS.register_module()
class UWMGITDataset(CustomDataset):
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 use_mosaic=False,
                 mosaic_center=(0.25, 0.75),
                 mosaic_prob=0.5,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend='disk'),
                 multi_label=False):
        super(UWMGITDataset, self).__init__(pipeline, img_dir, img_suffix, ann_dir, seg_map_suffix,
                                            split, data_root, test_mode, ignore_index, reduce_zero_label,
                                            classes, palette, gt_seg_map_loader_cfg, file_client_args)
        self.multi_label = multi_label
        self.use_mosaic = use_mosaic
        if self.use_mosaic:
            self.mosaic_prob = mosaic_prob
            self.mosaic_center = mosaic_center
            mosaic_at = [_['type'] == "Mosaic" for _ in pipeline].index(True)
            self.load_pipeline = Compose(pipeline[:mosaic_at])
            print(self.load_pipeline)
            self.pipeline = Compose(pipeline[mosaic_at + 1:])
        else:
            self.pipeline = Compose(pipeline)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        def random_crop(img, w, h):
            x = int(np.random.randint(0, img.shape[1] - w, 1))
            y = int(np.random.randint(0, img.shape[0] - h, 1))
            return x, x + w, y, y + h

        if self.use_mosaic and np.random.rand() < self.mosaic_prob:
            idxes = [idx] + np.random.randint(0, len(self), 3).tolist()
            results_list = []
            for idx in idxes:
                img_info = self.img_infos[idx]
                ann_info = self.get_ann_info(idx)
                results = dict(img_info=img_info, ann_info=ann_info)
                self.pre_pipeline(results)
                results_list.append(self.load_pipeline(results))
            results = results_list[0]
            img = results['img'].copy();
            img[:] = 0
            masks = {k: results[k].copy() * 0 + 255 for k in results['seg_fields']}
            center_x = int(round(np.random.uniform(*self.mosaic_center) * img.shape[1]))
            center_y = int(round(np.random.uniform(*self.mosaic_center) * img.shape[0]))
            for i in range(4):
                if i == 0:
                    x1, x2, y1, y2 = random_crop(results_list[i]['img'], center_x, center_y)
                    img[:center_y, :center_x] = results_list[i]['img'][y1:y2, x1:x2]
                    for k in masks:
                        masks[k][:center_y, :center_x] = results_list[i][k][y1:y2, x1:x2]
                elif i == 1:
                    x1, x2, y1, y2 = random_crop(results_list[i]['img'], img.shape[1] - center_x, center_y)
                    img[:center_y, center_x:] = results_list[i]['img'][y1:y2, x1:x2]
                    for k in masks:
                        masks[k][:center_y, center_x:] = results_list[i][k][y1:y2, x1:x2]
                elif i == 2:
                    x1, x2, y1, y2 = random_crop(results_list[i]['img'], center_x, img.shape[0] - center_y)
                    img[center_y:, :center_x] = results_list[i]['img'][y1:y2, x1:x2]
                    for k in masks:
                        masks[k][center_y:, :center_x] = results_list[i][k][y1:y2, x1:x2]
                elif i == 3:
                    x1, x2, y1, y2 = random_crop(results_list[i]['img'], img.shape[1] - center_x,
                                                 img.shape[0] - center_y)
                    img[center_y:, center_x:] = results_list[i]['img'][y1:y2, x1:x2]
                    for k in masks:
                        masks[k][center_y:, center_x:] = results_list[i][k][y1:y2, x1:x2]
            results['img'] = img
            for k in masks:
                results[k] = masks[k]
            return self.pipeline(results)
        elif self.use_mosaic:
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
            return self.pipeline(self.load_pipeline(results))
        else:
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
            return self.pipeline(results)

    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Format the results into dir (standard format for Cityscapes
        evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        result_files = []
        for res, idx in zip(results, indices):
            if len(res.shape) == 3 and not np.issubdtype(res.dtype, np.integer):
                result_file = osp.join(imgfile_prefix, self.img_infos[idx]["filename"][:-4] + ".png")
                if not osp.exists(osp.dirname(result_file)):
                    os.system(f"mkdir -p {osp.dirname(result_file)}")
                cv2.imwrite(result_file, (res * 65535).astype(np.uint16))
            else:
                result_file = osp.join(imgfile_prefix, self.img_infos[idx]["filename"][:-4] + ".png")
                if not osp.exists(osp.dirname(result_file)):
                    os.system(f"mkdir -p {osp.dirname(result_file)}")
                Image.fromarray(res.astype(np.uint8)).save(result_file)
            result_files.append(result_file)

        return result_files

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            if self.multi_label:
                ious = []
                for i in range(len(self.CLASSES)):
                    iou = intersect_and_union(
                        pred[..., i], seg_map[..., i], 2,
                        self.ignore_index, self.label_map,
                        self.reduce_zero_label)
                    ious.append(iou)
                ious = tuple([torch.stack([_[i] for _ in ious], 0)[:, 1] for i in range(len(ious[0]))])
                pre_eval_results.append(ious)
            else:
                pre_eval_results.append(
                    intersect_and_union(pred, seg_map, len(self.CLASSES),
                                        self.ignore_index, self.label_map,
                                        self.reduce_zero_label))

        return pre_eval_results

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset
        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'imDice', 'imIoU', 'imFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics.pop('fwIoU', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            elif key == 'fwIoU':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            if key == 'fwIoU':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
