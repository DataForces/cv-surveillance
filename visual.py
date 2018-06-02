#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Date: 2018-03-09T12:18:53+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
import time

import chainer
from chainercv.links import SSD300
from chainercv.visualizations import vis_bbox
from utils import pose_bbox_label_names
from utils import PoseBboxDataset


def img_prediction(split='trainval', has_anno=True):
    if not os.path.exists('visualization/imgs/{}'.format(split)):
        os.mkdir('visualization/imgs/{}'.format(split))

    dataset = PoseBboxDataset(split=split, return_id=True, has_anno=has_anno)
    start = time.time()
    if has_anno:
        for i in range(len(dataset)):
            ori_img, ori_bbox, ori_label, img_id = dataset[i]
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
            plt.savefig('visualization/imgs/{}/{}.png'.format(split, img_id))
            print("working on", img_id, '\r')
    else:
        for i in range(len(dataset)):
            ori_img, ori_bbox, ori_label, img_id = dataset[i]
            pred_bbox, pred_labels, pred_scores = model.predict([ori_img])
            plt.figure(dpi=120)
            vis_bbox(ori_img, pred_bbox[0], pred_labels[0], pred_scores[0],
                     label_names=label_names)
            plt.title("Prediction", fontsize=24)
            plt.savefig('visualization/imgs/{}/{}.png'.format(split, img_id))
            plt.close()
            print("working on", img_id, '\r')

    period = time.time() - start
    print('Number of Samples', len(dataset))
    print('Time comsuing:', period)
    print("Device FPS:", round(len(dataset) / period, 2))


def img_to_video(img_dir, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    imgfiles = glob.glob(os.path.join('visualization/imgs', img_dir, '*.png'))
    frame = cv2.imread(imgfiles[0])
    a, b = frame.shape[:2]
    imgsize = (b, a)
    save_file = 'visualization/videos/{}.mp4'.format(img_dir)
    print("saving video in ", save_file)
    out = cv2.VideoWriter(save_file, fourcc, fps, imgsize)
    for imgfile in imgfiles:
        frame = cv2.imread(imgfile)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument(
        '--pretrained', default='result/pose_iter_new_10000.npz')
    args = parser.parse_args()

    label_names = pose_bbox_label_names

    model = SSD300(
        n_fg_class=len(label_names),
        pretrained_model=args.pretrained)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # calculate fps
    img_prediction('sample', 'True')

    # img_splits = ['trainval','test','all-unmarked']
    # img_split_annos = [True, True, False]
    # for idx, split in enumerate(img_splits):
    #     img_prediction(split, img_split_annos[idx])
    #     img_to_video(split, 5)
