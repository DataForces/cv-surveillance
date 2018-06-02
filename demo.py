#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Date: 2018-03-09T12:18:53+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import matplotlib.pyplot as plt

import chainer
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox

from utils import pose_bbox_label_names
from utils import PoseBboxDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained', default='./models/model_iter_12000.npz')
    args = parser.parse_args()

    label_names = pose_bbox_label_names
    BboxDataset = PoseBboxDataset

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(label_names),
            pretrained_model=args.pretrained)
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(label_names),
            pretrained_model=args.pretrained)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    for i in range(0, 10):
        dataset = BboxDataset(split='test')
        ori_img, ori_bbox, ori_label = dataset[i]
        pred_bbox, pred_labels, pred_scores = model.predict([ori_img])

        fig = plt.figure(figsize=(20, 10), dpi=80)
        fig.suptitle("Original vs Prediction Annotations", fontsize=32)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xlabel("Original", fontsize=24)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xlabel("Prediction",fontsize=24)
        vis_bbox(ori_img, ori_bbox, ori_label,
            label_names=label_names, ax=ax1)
        vis_bbox(ori_img, pred_bbox[0], pred_labels[0], pred_scores[0],
            label_names=label_names, ax=ax2)
        plt.tight_layout()
        plt.savefig('visualization/compare-{}.png'.format(i))


if __name__ == '__main__':
    main()