import collections

from torch import optim
from eval_csv import evaluate
from model.fcos import FCOSDetector
import torch
from dataset.CSV_dataset import CSVDataset
import math, time
from dataset.augment import Transforms
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from utils.callbacks import LossHistory

print('CUDA available: {}'.format(torch.cuda.is_available()))

#指定GPU卡号
torch.cuda.set_device(1)



#单帧检测器的训练程序
def main(args=None):

    #运行命令中的参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--num_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument('--train_annots', default='datasets/Indices/2022_train.txt',
                        help='Path to file containing training annotations')
    parser.add_argument('--eval_annots', default='datasets/Indices/2022_eval.txt',
                        help='Path to file containing evaliating annotations')
    opt = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    
    #数据预处理
    transform = Transforms()
    #训练集加载程序
    train_dataset = CSVDataset(annot_file=opt.train_annots, resize_size=[320, 320], is_train=True, augment=transform)
    #测试集加载程序
    eval_dataset = CSVDataset(annot_file=opt.eval_annots, resize_size=[320, 320], is_train=True, augment=None)

    #网络模型——实际为论文中的FCST网络
    model = FCOSDetector(mode="train").cuda()
    model = torch.nn.DataParallel(model, device_ids=[1])

    #加载预训练权重
    # pretrained_net_dict = torch.load('checkpoint/backbone/TinyVovnet_list_320_ASFF-pretrain.pth')
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_net_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs

    #数据集加载器  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=opt.num_cpu, worker_init_fn=np.random.seed(0))

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                              collate_fn=eval_dataset.collate_fn)

    print("total_images : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE

    #优化算法——Adam 初始学习率1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    loss_hist = collections.deque(maxlen=50)

    model.train()

    #记录网络训练中的损失变化
    loss_history = LossHistory("logs/","TinyVovnet_320_ASFF")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = []

        for iter_step, data in enumerate(train_loader):
        
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            start_time = time.time()
            optimizer.zero_grad()
            
            #前向计算损失值
            losses = model([batch_imgs, batch_boxes, batch_classes])
            total_loss = losses[-1]
            
            #反向传播
            total_loss.mean().backward()
            optimizer.step()

            loss_hist.append(float(total_loss))
            epoch_loss.append(float(total_loss))

            #运行时间
            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            
            #日志打印
            print("epoch:%d iters:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms total_loss:%.4f" % \
                  (epoch + 1, iter_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(),
                   cost_time, np.mean(loss_hist)))

            if iter_step % 5==0:
                loss_history.append_loss_val(float(total_loss),losses[0].mean(),losses[1].mean(), losses[2].mean())

        print('Evaluating dataset')
        
        #每个Epoch后评估模型性能
        F1_score = evaluate(model, eval_loader, 'record/backbone/TinyVovnet_320_ASFF.txt')

        scheduler.step(np.mean(epoch_loss))

        #保存训练权重
        if epoch > EPOCHS / 2:
            torch.save(model.state_dict(), "checkpoint/backbone/TinyVovnet_320_ASFF_{}_F1-{}.pth".format(epoch + 1, F1_score))



if __name__ == '__main__':
    main()
