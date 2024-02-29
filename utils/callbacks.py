import os

import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, log_dir, backnone):
        import datetime
        curr_time       = datetime.datetime.now()
        time_str        = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        # self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.save_path  = os.path.join(self.log_dir, backnone, str(self.time_str))
        self.total_loss     = []
        self.cls_loss   = []
        self.cen_loss   = []
        self.reg_loss   = []

        os.makedirs(self.save_path)

    def append_loss(self, losses):
        self.total_loss.append(float(losses[-1]))
        self.cls_loss.append(losses[0].mean())
        self.cen_loss.append(losses[1].mean())
        self.reg_loss.append(losses[2].mean())

        with open(os.path.join(self.save_path, "epoch_total_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(float(losses[-1])))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_cls_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(float(losses[0].mean())))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_cen_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(float(losses[1].mean())))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_reg_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(float(losses[2].mean())))
            f.write("\n")

        self.loss_plot()


    def append_loss_val(self, totalloss,clsloss,cenloss,regloss):
        self.total_loss.append(totalloss)
        self.cls_loss.append(clsloss)
        self.cen_loss.append(cenloss)
        self.reg_loss.append(regloss)

        with open(os.path.join(self.save_path, "epoch_total_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(totalloss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_cls_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(clsloss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_cen_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(cenloss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_reg_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(regloss))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.total_loss))

        plt.figure()
        # plt.plot(iters, self.cls_loss, 'green', linestyle = '--',linewidth = 1, label='Classification loss')
        plt.plot(iters, self.cls_loss, 'green',linewidth = 1, label='Classification loss')
        plt.plot(iters, self.cen_loss, 'blue', linewidth = 1, label='Centerness loss')
        plt.plot(iters, self.reg_loss, 'yellow', linewidth = 1, label='Regression loss')

        # plt.grid(True)
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_multi" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")



        plt.figure()
        plt.plot(iters, self.total_loss, 'blue', linewidth=1, label='Total Loss')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_path, "epoch_train_loss" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")
