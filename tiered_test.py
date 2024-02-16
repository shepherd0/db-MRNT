import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from utils.local_generator import episode_local_generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.task_generator1 import mini_imagenet_folder, TaskGenerator
from torch.nn import init
from network_dil2 import MsCRN
import scipy as sp
import scipy.stats


parser = argparse.ArgumentParser(description="concat methods for few shot learning")
parser.add_argument("-w", "--way", type=int, default=5)
parser.add_argument("-s", "--shot", type=int, default=1)
parser.add_argument("-q", "--query", type=int, default=15)
parser.add_argument("-v", "--val", type=int, default=1)

parser.add_argument("-lr", "--init_learning_rate", type=float, default=0.001)

parser.add_argument("--size1", type=int, default=19)
parser.add_argument("--size2", type=int, default=14)
parser.add_argument("--size3", type=int, default=10)
parser.add_argument("--feature_dim", type=int, default=64)

parser.add_argument("--dataset", type=str, default="miniImagenet")
parser.add_argument("--channel", type=int, default=3)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-ti", "--test_times", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=500000)

parser.add_argument("--train_folder", type=str, default="E:/dataset/tieredImageNet/train")
parser.add_argument("--test_folder", type=str, default="E:/dataset/tieredImageNet/test")
parser.add_argument("--val_folder", type=str, default="E:/dataset/tieredImageNet/val")

parser.add_argument("--feature_encoder", type=str, default="Conv64")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--load_model", type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

CLASS_NUM = args.way
CLASS_SHOT = args.shot
CLASS_QUERY = args.query
CLASS_VAL = args.val

TRAIN_EPISODE = args.episode
TEST_EPISODE = args.test_episode
CHANNEL = args.channel
FEATURE_DIM = args.feature_dim

SIZE1 = args.size1
SIZE2 = args.size2
SIZE3 = args.size3
DIRECTORY = "INFO"

GPU = args.gpu
FEATURE_ENCODER = args.feature_encoder
INIT_LEARNING_RATE = args.init_learning_rate


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def test(model, task_generator):

    print("testing.......")
    total_accuracy = 0.0
    total_accuracy_f123 = 0.0
    with torch.no_grad():
        model.eval()
        for i in range(args.test_times):

            accuracies = []
            accuracies_f123 = []
            per_100_acc = 0.0
            per_100_acc_f123 = 0.0
            for test_episode in range(args.test_episode):

                support_x, support_y, query_x, query_y = task_generator.sample_task(args.way, args.shot,
                                                                                    args.query,
                                                                                    type="meta_test")
                support_x = support_x.to(device)  # (5,3,84,84)
                query_x = query_x.to(device)  # (75,5,84,84)

                local_act = episode_local_generator(support_y, query_y, CLASS_NUM, CLASS_SHOT)  # (75,5)
                fpn_score1, fpn_score2, fpn_score3, scores_0, scores_1, scores_2, scores_3, scores_4 = model(support_x, query_x, CLASS_NUM,
                                                                         CLASS_QUERY, FEATURE_DIM)

                # 直接or除以3
                scores_multi = scores_0 + scores_1 + scores_2 + scores_3 + scores_4
                f123 = (fpn_score1 + fpn_score2 + fpn_score3) / 3.0
                predict_labels_f123 = scores_multi + f123
                predict_labels = scores_multi + fpn_score1 + fpn_score2 + fpn_score3

                _, predict_labels_f123 = torch.max(predict_labels_f123.data, 1)
                _, predict_labels = torch.max(predict_labels.data, 1)
                _, local = torch.max(local_act.data, 1)

                rewards_f123 = predict_labels_f123.eq(local.to(device)).sum().item()
                rewards = predict_labels.eq(local.to(device)).sum().item()

                test_accuracy = rewards / 1.0 / (args.way * args.query)
                per_100_acc += test_accuracy
                accuracies.append(test_accuracy)

                test_accuracy_f123 = rewards_f123 / 1.0 / (args.way * args.query)
                per_100_acc_f123 += test_accuracy_f123
                accuracies_f123.append(test_accuracy_f123)

                if (test_episode + 1) % 100 == 0:
                    print("test_accuracy :", per_100_acc / 100, "test_accuracy_f123 :", per_100_acc_f123 / 100)
                    per_100_acc = 0.0
                    per_100_acc_f123 = 0.0

            test_accuracy, h = mean_confidence_interval(accuracies)
            test_accuracy_f123, h_f123 = mean_confidence_interval(accuracies_f123)

            total_accuracy += test_accuracy
            total_accuracy_f123 += test_accuracy_f123

            print("Time :", i + 1, "test accuracy:", test_accuracy, "h", h,
                  "test accuracy_f123:", test_accuracy_f123, "h", h_f123)
        print("avg_test_accuracy : ", total_accuracy / args.test_times)
        print("avg_test_accuracy_f123 : ", total_accuracy_f123 / args.test_times)


def main():
    # Step 1:init data folders
    print("init data folders")
    if args.dataset == "miniImagenet":
        metatrain_folder, metatval_folder, metatest_folder = mini_imagenet_folder(args.train_folder, args.val_folder, args.test_folder)
    elif args.dataset == "tiered":
        pass
    task_generator = TaskGenerator(metatrain_folder, metatval_folder, metatest_folder)

    # Step 2:init Model
    print("loading MsCRN Neural Network")
    model = MsCRN(args.way, args.shot, args.query, args.feature_encoder)
    model.load_state_dict(torch.load("./models/" +
                                     "Episode397500-test_accuracy-0.6145777777777778.pkl", map_location={'cuda:': 'cuda:' + str(args.gpu)}))
    print("load model ok!")
    model.to(device)
    # if args.train_embedding:
    #     embedding_train(model, task_generator)
    #     torch.cuda.empty_cache()
    # relation_train(model, task_generator)
    test(model, task_generator)


if __name__ == '__main__':
    main()
