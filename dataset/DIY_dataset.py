import torch
import math, random
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random

#高斯噪声
def GaussianNoise(img,boxes,mean=0,var=0.002):

    img = np.array(img/255,dtype=float)
    noise = np.random.normal(mean,var**0.5,img.shape)

    out = img + noise
    out = np.uint8(out*255)

    return out,boxes

#随机擦除
def RandomErasing(img, boxes, sl=0.01, sh=0.05, r1=0.5):

    attempts = random.randint(1,4)

    for attempt in range(attempts):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            y1 = random.randint(0, img.shape[0] - h)
            x1 = random.randint(0, img.shape[1] - w)

            cube = np.array([y1,y1 + h,x1,x1 + w])
            iou = iou_2d(cube,np.squeeze(boxes,axis=0))

            if iou < 0.001:
                img[y1:y1 + h, x1:x1 + w, 0] = torch.rand(h, w)
                img[y1:y1 + h, x1:x1 + w, 1] = torch.rand(h, w)
                img[y1:y1 + h, x1:x1 + w, 2] = torch.rand(h, w)

    return img, boxes

#计算IOU
def iou_2d(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / (float(boxAArea + boxBArea - interArea)+0.0001)

    return iou


#用于加载序列图像数据集的类
class DIYDataset(torch.utils.data.Dataset):

    CLASSES_NAME = ("__background__ ", "Target",)

    def __init__(self, video_list, resize_size=[256, 256],seq_len=8,skip=False,augment=None):

        self.name2id = dict(zip(DIYDataset.CLASSES_NAME, range(len(DIYDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize_size = resize_size
        #归一化参数
        self.mean = [0.42, 0.42, 0.42]
        self.std = [0.1, 0.1, 0.1]

        self.augment = augment
        self.seq_len = seq_len
        self.skip = skip

        self.file_names = list()
        with open(video_list) as f:
            videoPath_lists = f.readlines()
            self.num_videos = len(videoPath_lists)
            for video_path in videoPath_lists:
                self.file_names.append(video_path)

        print("INFO=====>List dataset init finished  ! !")
    #有多少个序列
    def __len__(self):
        return self.num_videos

    #取出一个序列中seq_len 长度的图像数据以及标注信息
    def __getitem__(self, i):

        annot_path = self.file_names[i].strip()+'/gt.txt'
        file_name_list,boxes_list,label_name_list = self.parse_annot_lines(annot_path)
        list_len = len(file_name_list)

        seq_img = list()
        seq_boxes = list()
        seq_classes = list()

        for idx in range(list_len):
            img = Image.open(file_name_list[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')

            boxes = boxes_list[idx]
            boxes = np.array(boxes, dtype=np.float32)
            label_names = label_name_list[idx]

            # if label_names[0]=='Targetw':
            #     print(file_name_list[0])

            classes = [self.name2id[name] for name in label_names]

            if self.augment is not None:
                img, boxes = self.augment(img, boxes)

            img = np.array(img)
            img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

            # if random.random() < 0.2:
            #     img, boxes = RandomErasing(img, boxes)
            # if random.random() < 0.2:
            #     img, boxes = GaussianNoise(img, boxes)

            img = transforms.ToTensor()(img)
            boxes = torch.from_numpy(boxes)
            classes = torch.LongTensor(classes)

            seq_img.append(img)
            seq_boxes.append(boxes)
            seq_classes.append(classes)

        return seq_img, seq_boxes, seq_classes

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

        seq_imgs_list, seq_boxes_list, seq_classes_list = zip(*data)

        bt_size = len(seq_imgs_list)
        seq_len = len(seq_imgs_list[0])

        imgs_list = [data for list in seq_imgs_list for data in list]
        boxes_list = [data for list in seq_boxes_list for data in list]
        classes_list = [data for list in seq_classes_list for data in list]

        # imgs_list, boxes_list, classes_list = zip(*data)

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


        batch_list_imgs = list()
        batch_list_boxs = list()
        batch_list_classes = list()
        for i in range(bt_size):
            seq_imgs = list()
            seq_boxs = list()
            seq_classes = list()
            for j in range(seq_len):
                seq_imgs.append(pad_imgs_list[i*seq_len+j])
                seq_boxs.append(pad_boxes_list[i*seq_len+j])
                seq_classes.append(pad_classes_list[i*seq_len+j])
            batch_list_imgs.append(torch.stack(seq_imgs, 0))
            batch_list_boxs.append(torch.stack(seq_boxs, 0))
            batch_list_classes.append(torch.stack(seq_classes, 0))

        batch_seq_imgs = torch.stack(batch_list_imgs, 0)
        batch_seq_boxs = torch.stack(batch_list_boxs, 0)
        batch_seq_classes = torch.stack(batch_list_classes, 0)

        return batch_seq_imgs, batch_seq_boxs, batch_seq_classes

    #解析标注文件中的一行  如：dataset/images/aaa.png 23 33 62 67 Target
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

        video_size = len(file_name_list)

        if self.skip:
            skip = random.randint(1, int(video_size / self.seq_len))
            start = random.randint(0, video_size - self.seq_len * skip)
        else:
            start = np.random.randint(video_size - self.seq_len)


        clip_boxes_list = boxes_list[start:self.seq_len+start]
        clip_label_name_list = label_name_list[start:self.seq_len+start]
        clip_file_name_list = file_name_list[start:self.seq_len+start]

        return  clip_file_name_list,clip_boxes_list,clip_label_name_list







if __name__ == "__main__":
    pass
    eval_dataset = DIYDataset(root_dir='datasets', resize_size=[256, 256],
                              split='test', use_difficult=False, is_train=False, augment=None)
    print(len(eval_dataset.CLASSES_NAME))
    dataset = DIYDataset("datasets", split='trainval')
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
