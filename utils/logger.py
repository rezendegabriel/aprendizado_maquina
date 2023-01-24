#%% LIBRARIES

from matplotlib import pyplot as plt

import matplotlib as mpl
import numpy as np
import os

#%% CLASS

class Logger:
    def __init__(self):
        self.loss_train = []
        self.loss_val = []

        self.acc_train = []
        self.acc_val = []

        self.class_weights = []
        self.sampler_weights = []

    def get_logs(self):
        return self.loss_train, self.loss_val, self.acc_train, self.acc_val, self.class_weights, self.sampler_weights

    def restore_logs(self, logs):
        self.loss_train, self.loss_val, self.acc_train, self.acc_val, self.class_weights, self.sampler_weights = logs

    def save_plt(self, hps, k):
        # Hyper-parameters
        model_save_dir = hps["model_save_dir"]
        num_epochs = hps["num_epochs"]
        num_outputs = hps["num_outputs"]
        z = hps["z"]

        if z != 0:
            x_s = np.arange(0, num_epochs, z)

        acc_path = os.path.join(model_save_dir, "acc_{}-fold.jpg".format(k))
        loss_path = os.path.join(model_save_dir, "loss_{}-fold.jpg".format(k))
        class_weights_plot_path = os.path.join(model_save_dir, "class_weights_{}-fold.jpg".format(k))
        sampler_weights_plot_path = os.path.join(model_save_dir, "sampler_weights_{}-fold.jpg".format(k))

        mpl.style.use("seaborn")
        
        # Accuracy
        plt.figure(figsize = (40, 40))
        plt.plot(self.acc_train, label = "Training Accuracy")
        plt.plot(self.acc_val, label = "Validation Accuracy")
        
        plt.xlim(1, num_epochs)
        plt.ylim(0, 100)
        plt.grid(True)
        plt.xlabel("Epochs", labelpad = 72, fontsize = 72)
        plt.xticks(fontsize = 60)
        plt.ylabel("Accuracy", labelpad = 72, fontsize = 72)
        plt.yticks(fontsize = 60)
        plt.legend(loc = "best", fontsize = 72, frameon = True)

        plt.savefig(acc_path, bbox_inches = "tight", pad_inches = 0)

        # Loss
        plt.figure(figsize = (40, 40))
        plt.plot(self.loss_train, label = "Training Loss")
        plt.plot(self.loss_val, label = "Validation Loss")
        
        plt.xlim(1, num_epochs)
        plt.grid(True, color = "white")
        plt.xlabel("Epochs", labelpad = 72, fontsize = 72)
        plt.xticks(fontsize = 60)
        plt.ylabel("Loss", labelpad = 72, fontsize = 72)
        plt.yticks(fontsize = 60)
        plt.legend(loc = "best", fontsize = 72, frameon = True)

        plt.savefig(loss_path, bbox_inches = "tight", pad_inches = 0)

        if z != 0:
            mpl.style.use("default")

            # Weights (bar)
            class_weights = []
            sampler_weights = []
            
            labels = [l for l in range(num_outputs)]

            for c in range(len(labels)):
                class_weights.append([])
                sampler_weights.append([])

            for w in self.class_weights:
                for c_w, class_weight in enumerate(w):
                    class_weights[c_w].append(class_weight)

            for w in self.sampler_weights:
                for s_w, sampler_weight in enumerate(w):
                    sampler_weights[s_w].append(sampler_weight)

            colors = ["red", "orange", "yellow", "green", "blue", "indigo", "darkviolet"]

            # Class weights plot
            fig, ax = plt.subplots(1, figsize = (10, 10))

            bottom = len(class_weights[0])*[0]
            for c in range(len(labels)):
                if z == 1:
                    plt.bar(x_s, class_weights[c], width = 1, bottom = bottom, color = colors[c])
                else:
                    plt.bar(x_s, class_weights[c], width = 20, bottom = bottom, color = colors[c])

                bottom = [b + w for b, w in zip(bottom, class_weights[c])]

            plt.legend(labels, loc = "best", frameon = True)

            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)

            plt.xlabel("Epochs")
            if z == 1:
                xticks = x_s
                xlabels = ["" for i in x_s]
                plt.xticks(xticks, xlabels)
                plt.xlim(-0.5, ax.get_xticks()[-1]+0.5)
            else:
                xticks = x_s
                xlabels = ["{}".format(i) for i in x_s]
                plt.xticks(xticks, xlabels)
                plt.xlim(-10, ax.get_xticks()[-1]+10)

            plt.ylabel("Weights")
            yticks = np.arange(0, 1.1, .1)
            ylabels = ["{}%".format(i) for i in np.arange(0, 101, 10)]
            plt.yticks(yticks, ylabels)
            ax.yaxis.grid(color = "gray", linestyle = "dashed")

            plt.savefig(class_weights_plot_path, bbox_inches = "tight", pad_inches = 0)

            # Sampler weights plot
            fig, ax = plt.subplots(1, figsize = (10, 10))

            bottom = len(sampler_weights[0])*[0]
            for c in range(len(labels)):
                if z == 1:
                    plt.bar(x_s, sampler_weights[c], width = 1, bottom = bottom, color = colors[c])
                else:
                    plt.bar(x_s, sampler_weights[c], width = 20, bottom = bottom, color = colors[c])

                bottom = [b + w for b, w in zip(bottom, sampler_weights[c])]

            plt.legend(labels, loc = "best", frameon = True)

            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)

            plt.xlabel("Epochs")
            if z == 1:
                xticks = x_s
                xlabels = ["" for i in x_s]
                plt.xticks(xticks, xlabels)
                plt.xlim(-0.5, ax.get_xticks()[-1]+0.5)
            else:
                xticks = x_s
                xlabels = ["{}".format(i) for i in x_s]
                plt.xticks(xticks, xlabels)
                plt.xlim(-10, ax.get_xticks()[-1]+10)

            plt.ylabel("Weights")
            yticks = np.arange(0, 1.1, .1)
            ylabels = ["{}%".format(i) for i in np.arange(0, 101, 10)]
            plt.yticks(yticks, ylabels)
            ax.yaxis.grid(color = "gray", linestyle = "dashed")

            plt.savefig(sampler_weights_plot_path, bbox_inches = "tight", pad_inches = 0)