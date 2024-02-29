import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random

class SeqDataset(torch.utils.data.Dataset):

    CLASSES_NAME = ("__background__ ", "Target",)

    def __init__(self, video_list, resize_size=[256, 256], augment=None):

        self.name2id = dict(zip(SeqDataset.CLASSES_NAME, range(len(SeqDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize_size = resize_size
        self.mean = [0.42, 0.42, 0.42]
        self.std = [0.1, 0.1, 0.1]
        self.augment = augment

        self.file_names = list()
        with open(video_list) as f:
            videoPath_lists = f.readlines()
            self.num_videos = len(videoPath_lists)
            for video_path in videoPath_lists:
                self.file_names.append(video_path)

        print("INFO=====>IRSTD dataset init finished  ! !")

    def __len__(self):
        return self.num_videos

    def __getitem__(self, i):

        annot_path = self.file_names[i].strip()+'/gt.txt'
        file_name_list,boxes_list,label_name_list = self.parse_annot_lines(annot_path)
        seq_len = len(file_name_list)

        seq_img = list()
        seq_boxes = list()
        seq_classes = list()

        for idx in range(seq_len):
            img = Image.open(file_name_list[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            boxes = boxes_list[idx]
            boxes = np.array(boxes, dtype=np.float32)
            label_names = label_name_list[idx]
            classes = [self.name2id[name] for name in label_names]
            img = np.array(img)
            img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize(self.mean, self.std, inplace=True)(img)
            boxes = torch.from_numpy(boxes)
            classes = torch.LongTensor(classes)

            seq_img.append(img)
            seq_boxes.append(boxes)
            seq_classes.append(classes)

        return seq_img,seq_boxes,seq_classes

    def parse_annot_lines(self,annot_path):
        boxes_list = []
        label_name_list = []
        file_name_list = []
        # txt with img_path x1 y1 x2 y2 class_name...
        with open(annot_path) as f:
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split(',')
                file_name = splited[0]
                file_name = file_name.replace('data/images','datasets/list_imgs')
                file_name_list.append(file_name)
                num_boxes = (len(splited) - 1) // 5
                box = []
                label = []
                for i in range(num_boxes):
                    xmin = splited[1 + 5 * i]
                    ymin = splited[2 + 5 * i]
                    xmax = splited[3 + 5 * i]
                    ymax = splited[4 + 5 * i]
                    class_name = splited[5 + 5 * i]
                    box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                    label.append(class_name)
                boxes_list.append(box)
                label_name_list.append(label)

        return  file_name_list,boxes_list,label_name_list

    def preprocess_img_boxes(self, image, boxes, input_ksize):

        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 0
        pad_h = 0

        if nw % 16 != 0:
            pad_w = 16 - nw % 16
        if nh % 16 != 0:
            pad_h = 16 - nh % 16

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes


