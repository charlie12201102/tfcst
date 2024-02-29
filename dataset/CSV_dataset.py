import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random


#图像翻转
def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes

#单帧图像数据集加载类
class CSVDataset(torch.utils.data.Dataset):
    CLASSES_NAME = ("__background__ ", "Target",)

    def __init__(self, annot_file, resize_size=[256, 256], is_train=True, augment=None):

        self.name2id = dict(zip(CSVDataset.CLASSES_NAME, range(len(CSVDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize_size = resize_size
        #归一化参数
        self.mean = [0.42, 0.42, 0.42]
        self.std = [0.1, 0.1, 0.1]
        self.train = is_train
        #是否采用数据增强
        self.augment = augment
        
        #图像文件路径
        self.file_names = []
        #检测框
        self.boxes = []
        #类别标签
        self.label_names = []
        # txt with img_path x1 y1 x2 y2 class_name...
        with open(annot_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)
            for line in lines:
                splited = line.strip().split()
                self.file_names.append(splited[0])
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
                self.boxes.append(box)
                self.label_names.append(label)

        print("INFO=====>IRSTD dataset init finished  ! !")

    #数据量
    def __len__(self):
        return self.num_samples

    #从数据集中取出指定idx的图像数据
    def __getitem__(self, idx):
        img = Image.open(self.file_names[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = self.boxes[idx]
        boxes = np.array(boxes, dtype=np.float32)
        label_names = self.label_names[idx]
        classes = [self.name2id[name] for name in label_names]
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.augment is not None:
                img, boxes = self.augment(img, boxes)
        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)
        #返回图像数据、检测框、类别标签
        return img, boxes, classes

    #解析一行  如：dataset/images/aaa.png 23 33 62 67 Target
    def parse_annot_lines(self, index):
        result = {}
        annot_line = self.annot_lines[index]

    #图像预处理——缩放到指定大小、加padding
    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
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

    #多幅图像数据包装
    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num: max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes

#测试
if __name__ == "__main__":
    pass
    eval_dataset = CSVDataset(root_dir='datasets', resize_size=[256, 256],
                              split='test', use_difficult=False, is_train=False, augment=None)
    print(len(eval_dataset.CLASSES_NAME))
    dataset = CSVDataset("datasets", split='trainval')
    for i in range(100):
        img, boxes, classes = dataset[i]
        img, boxes, classes = img.numpy().astype(np.uint8), boxes.numpy(), classes.numpy()
        img = np.transpose(img, (1, 2, 0))
        print(img.shape)
        print(boxes)
        print(classes)
        for box in boxes:
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            img = cv2.rectangle(img, pt1, pt2, [0, 255, 0], 3)
        cv2.imshow("test", img)
        if cv2.waitKey(0) == 27:
            break
    imgs, boxes, classes = eval_dataset.collate_fn([dataset[105], dataset[101], dataset[200]])
    print(boxes, classes, "\n", imgs.shape, boxes.shape, classes.shape, boxes.dtype, classes.dtype, imgs.dtype)
    for index, i in enumerate(imgs):
        i = i.numpy().astype(np.uint8)
        i = np.transpose(i, (1, 2, 0))
        i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        print(i.shape, type(i))
        cv2.imwrite(str(index) + ".jpg", i)
