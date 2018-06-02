import argparse
import copy
import os
import warnings
import numpy as np
import xml.etree.ElementTree as ET

import chainer
import chainer.training.extensions
from chainer import reporter
from chainercv.utils import read_image
from chainercv.utils import apply_prediction_to_iterator
from chainercv.evaluations import eval_detection_voc
from chainercv.visualizations import vis_bbox
import matplotlib.pyplot as plt


pose_bbox_label_names = (
    'sitting',
    'standing',
    'lying',
)


class PoseBboxDataset(chainer.dataset.DatasetMixin):

    """Bounding box dataset for Depth dataset
    modified from chainercv.datasets.VOCBboxDataset
    `img, bbox, label`: a tuple of an image, bounding boxes and labels.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'trainval', 'test'}): Select a split of the
            dataset.
        return_id (bool): return the id of the image or not.
        has_anno (bool): whether an image has corresponding Annotations.
    """

    def __init__(self, data_dir='./data/Depth', split='trainval', return_id=False, has_anno=True):
        self.data_dir = data_dir
        self.split = split
        if self.split not in ['trainval', 'test']:
            warnings.warn(
                'only [trainval, test] are available.'
            )
        id_list_file = os.path.join(
            self.data_dir, 'ImageSets/Main/{0}.txt'.format(self.split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.return_id = return_id
        self.has_anno = has_anno

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes (img_id)

        """
        id_ = self.ids[i]
        if self.has_anno:
            anno = ET.parse(
                os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
            bbox = list()
            label = list()

            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')
                # subtract 1 to make pixel indexes 0-based
                bbox.append([
                    int(bndbox_anno.find(tag).text) - 1
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
                name = obj.find('name').text.lower().strip()
                label.append(pose_bbox_label_names.index(name))
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = None
            label = None
        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.png')
        img = read_image(img_file, color=True)
        if self.return_id:
            return img, bbox, label, id_
        return img, bbox, label


class DetectionEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a detection model by PASCAL VOC metric.

    This extension iterates over an iterator and evaluates the prediction
    results by average precisions (APs) and mean of them
    (mean Average Precision, mAP).
    This extension reports the following values with keys.
    Please note that :obj:`'ap/<label_names[l]>'` is reported only if
    :obj:`label_names` is specified.

    * :obj:`'map'`: Mean of average precisions (mAP).
    * :obj:`'ap/<label_names[l]>'`: Average precision for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        For example, this evaluator reports :obj:`'ap/aeroplane'`, \
        :obj:`'ap/bicycle'`, etc. if :obj:`label_names` is \
        :obj:`~chainercv.datasets.voc_bbox_label_names`. \
        If there is no bounding box assigned to class :obj:`label_names[l]` \
        in either ground truth or prediction, it reports :obj:`numpy.nan` as \
        its average precision. \
        In this case, mAP is computed without this class.

    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, bbox, label` or
            :obj:`img, bbox, label, difficult`.
            :obj:`img` is an image, :obj:`bbox` is coordinates of bounding
            boxes, :obj:`label` is labels of the bounding boxes and
            :obj:`difficult` is whether the bounding boxes are difficult or
            not. If :obj:`difficult` is returned, difficult ground truth
            will be ignored from evaluation.
        target (chainer.Link): A detection link. This link must have
            :meth:`predict` method that takes a list of images and returns
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
        label_names (iterable of strings): An iterable of names of classes.
            If this value is specified, average precision for each class is
            also reported with the key :obj:`'ap/<label_names[l]>'`.

    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, use_07_metric=False, label_names=None):
        super(DetectionEvaluator, self).__init__(
            iterator, target)
        self.use_07_metric = use_07_metric
        self.label_names = label_names

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        del imgs

        pred_bboxes, pred_labels, pred_scores = pred_values

        if len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_difficults = gt_values
        elif len(gt_values) == 2:
            gt_bboxes, gt_labels = gt_values
            gt_difficults = None

        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=self.use_07_metric)

        report = {'map': result['map']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation


def remove_invalid_records(split='trainval'):
    dataset = PoseBboxDataset(split=split)
    all_ids = dataset.ids
    valid_ids = []
    invalid_ids = []
    for i in range(len(dataset)):
        try:
            img, bbox, label = dataset[i]
            if len(bbox) == 0:
                invalid_ids.append(all_ids[i])
            else:
                valid_ids.append(all_ids[i])
        except:
            invalid_ids.append(all_ids[i])
    assert len(invalid_ids)+len(valid_ids)== len(all_ids), "Number of ids should be consistent."
    print("Valid IDs in {} : {}".format(split, len(valid_ids)))
    valid_id_file = os.path.join(
        dataset.data_dir, 'ImageSets/Main/{0}-marked.txt'.format(split))
    with open(valid_id_file, "w") as f:
        f.write("\n".join(valid_ids))

    print("InValid IDs in {} : {}".format(split, len(invalid_ids)))
    invalid_id_file = os.path.join(
        dataset.data_dir, 'ImageSets/Main/{0}-unmarked.txt'.format(split))
    with open(invalid_id_file, "w") as f:
        f.write("\n".join(invalid_ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='old')
    args = parser.parse_args()
    # remove invalid records in txt file
    remove_invalid_records(args.split)

    dataset = PoseBboxDataset(split='trainval')
    img, bbox, label = dataset[0]
    vis_bbox(img, bbox, label, label_names=pose_bbox_label_names)
    plt.show()