import os
import math
import time
import argparse

import timm.scheduler
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

# parser.add_argument("--size1", type=int, default=19)
# parser.add_argument("--size2", type=int, default=14)
# parser.add_argument("--size3", type=int, default=10)
parser.add_argument("--feature_dim", type=int, default=64)

parser.add_argument("--dataset", type=str, default="miniImagenet")
parser.add_argument("--channel", type=int, default=3)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-e", "--episode", type=int, default=500000)

parser.add_argument("--train_folder", type=str, default="../datas/miniImagenet/train")
parser.add_argument("--test_folder", type=str, default="../datas/miniImagenet/test")
parser.add_argument("--val_folder", type=str, default="../datas/miniImagenet/val")

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

# SIZE1 = args.size1
# SIZE2 = args.size2
# SIZE3 = args.size3
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


def train(model, task_generator):

    F_txt = open(DIRECTORY + "/training_info_" + str(CLASS_NUM) + "way_" + str(CLASS_SHOT) + "shot.txt", 'a')
    if args.load_model == 1:
        if os.path.exists(str("./models/" + "20" + ".pkl")):
            model.load_state_dict(
                torch.load("./models/" + "20" + ".pkl", map_location={'cuda:': 'cuda:' + str(GPU)}))
            print("load model ok!", file=F_txt)
            print("load model ok!")
    else:
        print("Start train from the begin.......", file=F_txt)
        print("Start train from the begin.......")
    # print("multi_weight:", model.multi_w)
    # print("multi_weight:", model.multi_w, file=F_txt)

    # optim set
    optim = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
    # scheduler = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=2,
    #                               verbose=True)
    # optim = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=1e-4)
    # schedule = MultiStepLR(optim, [args.drop1, args.drop2, args.drop3], gamma=0.1)
    # optim = torch.optim.SGD(model.parameters(), lr=INIT_LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optim, step_size=150000, gamma=0.5)
    """
    """
    # 0.01,0.005,0.0025,0.00125,0.000625,0.0003125
    # 0    9w    18w    27w     36w      45w
    # optim = torch.optim.SGD(params=model.parameters(), lr=INIT_LEARNING_RATE)
    # scheduler = timm.scheduler.CosineLRScheduler(optimizer=optim,
    #                                              t_initial=500000,
    #                                              lr_min=0.0001,
    #                                              warmup_t=5000,
    #                                              warmup_lr_init=0.01
    #                                              )

    accuracy_list = []
    loss_list = []
    val_accuracies_list = []
    test_accuracies_list = []
    xtrain = []
    xloss = []
    xval = []
    xtest = []

    # mark point set
    total_rewards = 0
    max_accuracy = 0
    max_episode = 0
    last_val_accuracy = 0
    base_val_accuracy = 0.58
    max_val_accuracy = 0
    max_test_accuracy = 0
    max_test_accuracy_episode = 0
    max_val_accuracy_episode = 0
    base_accuracy = 0.58
    last_accuracy = 0
    T = 7
    alpha = 0.3
    # train process
    for train_episode in range(0, TRAIN_EPISODE):
        model.train()
        scheduler.step(train_episode)

        # init dataset
        support_x, support_y, query_x, query_y = task_generator.sample_task(args.way, args.shot, args.query,
                                                                            type="meta_train")
        support_x = support_x.to(device)  # (5,3,84,84)
        query_x = query_x.to(device)  # (75,5,84,84)

        local_act = episode_local_generator(support_y, query_y, CLASS_NUM, CLASS_SHOT)  # (75,5)
        fpn_score1, fpn_score2, fpn_score3, scores_0, scores_1, scores_2, scores_3, scores_4 = model(support_x, query_x, CLASS_NUM, CLASS_QUERY, FEATURE_DIM)  # (5 * 75, 5)

        # loss set and optim
        criterion = nn.CrossEntropyLoss()
        # criterion_ditillation = nn.KLDivLoss(reduction='batchmean')
        local_act = local_act.float().to(device)

        scores_fpn = fpn_score1 + fpn_score2 + fpn_score3  # S
        scores_multi = scores_0 + scores_1 + scores_2 + scores_3 + scores_4  # T

        loss0 = criterion(scores_0, local_act)
        loss1 = criterion(scores_1, local_act)
        loss2 = criterion(scores_2, local_act)
        loss3 = criterion(scores_3, local_act)
        loss4 = criterion(scores_4, local_act)

        lossf1 = criterion(fpn_score1, local_act)
        lossf2 = criterion(fpn_score2, local_act)
        lossf3 = criterion(fpn_score3, local_act)
        lossplus = lossf1 + lossf2 + lossf3 + loss0 + loss1 + loss2 + loss3 + loss4

        # lossS = criterion(scores_fpn, local_act)
        # ditillation_loss = criterion_ditillation(F.softmax(scores_fpn/T, dim=1), F.softmax(scores_multi/T, dim=1))

        loss_all = lossplus
        # loss_all = lossS * alpha + ditillation_loss * (1 - alpha) + lossplus
        # FPN视为单loss
        # scores_fpn = (fpn_score1 + fpn_score2 + fpn_score3) / 3.0
        # loss1fpn = criterion(scores_fpn, local_act)
        # # loss_multi = criterion(scores_multi, local_act)
        # loss = loss1fpn + loss0 + loss1 + loss2 + loss3 + loss4

        model.zero_grad()
        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()

        # train monitor
        # predict_labels = scores_0 + scores_1 + scores_2 + scores_3 + scores_4 + fpn_score1 + fpn_score2 + fpn_score3
        predict_labels = scores_multi + scores_fpn
        _, predict_labels = torch.max(predict_labels.data, 1)
        _, local_act = torch.max(local_act.data, 1)
        rewards = predict_labels.eq(local_act.to(device)).sum().item()

        # 检查每个得分的准确率
        # _, scores_0 = torch.max(scores_0.data, 1)
        # _, scores_1 = torch.max(scores_1.data, 1)
        # _, scores_2 = torch.max(scores_2.data, 1)
        # _, scores_3 = torch.max(scores_3.data, 1)
        # _, scores_4 = torch.max(scores_4.data, 1)
        # _, scores_f1 = torch.max(fpn_score1.data, 1)
        # _, scores_f2 = torch.max(fpn_score2.data, 1)
        # _, scores_f3 = torch.max(fpn_score3.data, 1)
        #
        # rewards0 = scores_0.eq(local_act.to(device)).sum().item()
        # rewards1 = scores_1.eq(local_act.to(device)).sum().item()
        # rewards2 = scores_2.eq(local_act.to(device)).sum().item()
        # rewards3 = scores_3.eq(local_act.to(device)).sum().item()
        # rewards4 = scores_4.eq(local_act.to(device)).sum().item()
        # rewardsf1 = scores_f1.eq(local_act.to(device)).sum().item()
        # rewardsf2 = scores_f2.eq(local_act.to(device)).sum().item()
        # rewardsf3 = scores_f3.eq(local_act.to(device)).sum().item()

        total_rewards += rewards
        if (train_episode + 1) % 100 == 0:
            accuracy = total_rewards / 1.0 / (args.way * args.query) / 100
            xtrain.append((train_episode + 1) / 100)
            accuracy_list.append(accuracy * 100)
            xloss.append((train_episode + 1) / 100)
            loss_list.append(loss_all.item())
            total_rewards = 0

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_episode = train_episode + 1

            print("train episode ", train_episode + 1, "train_accuracy: ", accuracy, "loss = ", loss_all.item(),
                  "   ***max***: ", max_accuracy, "max_episode_", max_episode, "keep_",
                  train_episode + 1 - max_episode, file=F_txt)
            print("train episode ", train_episode + 1, "train_accuracy: ", accuracy, "loss = ", loss_all.item(),
                  "   ***max***: ", max_accuracy, "max_episode_", max_episode, "keep_",
                  train_episode + 1 - max_episode)
        # accuracy = 0

        # eval and choose the best models
        if (train_episode + 1) % 2 == 0:
            model.eval()
            print("Validation...........", file=F_txt)
            print("Validation...........")
            accuracies = []
            accuracies_f123 = []
            with torch.no_grad():
                for i in range(TEST_EPISODE):
                    support_x, support_y, query_x, query_y = task_generator.sample_task(args.way, args.shot,
                                                                                        args.query,
                                                                                        type="meta_val")
                    support_x = support_x.to(device)  # (5,3,84,84)
                    query_x = query_x.to(device)  # (75,5,84,84)

                    local_act = episode_local_generator(support_y, query_y, CLASS_NUM, CLASS_SHOT)  # (75,5)
                    fpn_score1, fpn_score2, fpn_score3, scores_0, scores_1, scores_2, scores_3, scores_4 = model(support_x, query_x, CLASS_NUM,
                                                                                         CLASS_QUERY, FEATURE_DIM)

                    # 直接or除以3
                    f123 = (fpn_score1 + fpn_score2 + fpn_score3) / 3.0
                    scores_multi = scores_0 + scores_1 + scores_2 + scores_3 + scores_4

                    # 预测指标
                    predict_labels = scores_multi + fpn_score1 + fpn_score2 + fpn_score3
                    predict_labels_f123 = scores_multi + f123

                    _, predict_labels = torch.max(predict_labels.data, 1)
                    _, predict_labels_f123 = torch.max(predict_labels_f123.data, 1)
                    _, local = torch.max(local_act.data, 1)

                    # 得分计算
                    rewards = predict_labels.eq(local.to(device)).sum().item()
                    rewards_f123 = predict_labels_f123.eq(local.to(device)).sum().item()

                    # 准确率计算
                    val_accuracy = rewards / 1.0 / (args.way * args.query)
                    accuracies.append(val_accuracy)
                    val_accuracy_f123 = rewards_f123 / 1.0 / (args.way * args.query)
                    accuracies_f123.append(val_accuracy_f123)

                # 检查每个得分
                _, scores_multi = torch.max(scores_multi.data, 1)
                _, scores_f1 = torch.max(fpn_score1.data, 1)
                _, scores_f2 = torch.max(fpn_score2.data, 1)
                _, scores_f3 = torch.max(fpn_score3.data, 1)
                _, scores_0 = torch.max(scores_0.data, 1)
                _, scores_1 = torch.max(scores_1.data, 1)
                _, scores_2 = torch.max(scores_2.data, 1)
                _, scores_3 = torch.max(scores_3.data, 1)
                _, scores_4 = torch.max(scores_4.data, 1)
                _, scores_fpn = torch.max(f123.data, 1)

                rewardsm = scores_multi.eq(local.to(device)).sum().item()
                rewardsf1 = scores_f1.eq(local.to(device)).sum().item()
                rewardsf2 = scores_f2.eq(local.to(device)).sum().item()
                rewardsf3 = scores_f3.eq(local.to(device)).sum().item()
                rewards0 = scores_0.eq(local.to(device)).sum().item()
                rewards1 = scores_1.eq(local.to(device)).sum().item()
                rewards2 = scores_2.eq(local.to(device)).sum().item()
                rewards3 = scores_3.eq(local.to(device)).sum().item()
                rewards4 = scores_4.eq(local.to(device)).sum().item()
                rewardstotal = scores_fpn.eq(local.to(device)).sum().item()

                # print("check per score rewards of multi:")
                # print("check per score rewards of multi:", file=F_txt)

                print("score_multi:", rewardsm, "score_21:", rewards0, "score_10:", rewards1, "score_19:", rewards2,
                      "score_15: ", rewards3, "score_17:", rewards4)
                print("score_multi:", rewardsm, "score_21:", rewards0, "score_10:", rewards1, "score_19:", rewards2,
                      "score_15:", rewards3, "score_17:", rewards4, file=F_txt)

                print("check per score rewards of FPN:")
                print("check per score rewards of FPN:", file=F_txt)

                print("scoref1:", rewardsf1, "scoref2:", rewardsf2, "scoref3:",
                      rewardsf3, "scoreftotal: ", rewardstotal)
                print("scoref1:", rewardsf1, "scoref2:", rewardsf2, "scoref3:",
                      rewardsf3, "scoreftotal:", rewardstotal, file=F_txt)

                val_accuracy, h = mean_confidence_interval(accuracies)
                val_accuracy_f123, h_f123 = mean_confidence_interval(accuracies_f123)

                val_accuracies_list.append(val_accuracy_f123 * 100)
                xval.append((train_episode + 1) / 100)
                # scheduler.step(val_accuracy_plus)

                if val_accuracy_f123 > max_val_accuracy:
                    max_val_accuracy = val_accuracy_f123
                    max_val_accuracy_episode = train_episode + 1

                print("Four indicators:", file=F_txt)
                print("Four indicators:")

                print("val accuracy:", val_accuracy, "h", h, "max_val_accuracy", max_val_accuracy, "episode",
                      max_val_accuracy_episode, file=F_txt)
                print("val accuracy:", val_accuracy, "h", h, "max_val_accuracy", max_val_accuracy, "episode",
                      max_val_accuracy_episode)

                print("val_accuracy_f123:", val_accuracy_f123, "h_f123", h_f123)
                print("val_accuracy_f123:", val_accuracy_f123, "h_f123", h_f123, file=F_txt)

                if val_accuracy_f123 > last_val_accuracy:
                    # save networks
                    torch.save(model.state_dict(),
                               "./models/" + args.feature_encoder + "-" + str(args.dataset) + "-episode" + str(
                                   train_episode + 1)
                               + "-" + str(args.way) + "-way-" + str(args.shot) + "-shot-"
                               + "train_accuracy-" + str(accuracy) + "val_accuracy-" + str(val_accuracy_f123) + ".pkl")
                    print("save networks for episode:", train_episode + 1, file=F_txt)
                    print("save networks for episode:", train_episode + 1)
                    last_val_accuracy = val_accuracy_f123

                if val_accuracy_f123 > base_val_accuracy:
                    # save networks
                    torch.save(model.state_dict(),
                               "./models/check_point/" + "Episode" + str(
                                   train_episode + 1)
                               + "-" + str(args.way) + "-way-" + str(args.shot) + "-shot-"
                               + "train_accuracy-" + str(accuracy) + "val_accuracy-" + str(val_accuracy_f123) + ".pkl")
                    print("save networks for episode:", train_episode + 1, file=F_txt)
                    print("save networks for episode:", train_episode + 1)

        if (train_episode + 1) % 2 == 0:
            model.eval()
            print("Testing...........", file=F_txt)
            print("Testing...........")
            accuracies = []
            accuracies_f123 = []
            # print("multi_w:", model.multi_w)
            # print("multi_w:", model.multi_w, file=F_txt)
            with torch.no_grad():
                for i in range(TEST_EPISODE):
                    support_x, support_y, query_x, query_y = task_generator.sample_task(args.way, args.shot,
                                                                                        args.query,
                                                                                        type="meta_test")
                    support_x = support_x.to(device)  # (5,3,84,84)
                    query_x = query_x.to(device)  # (75,5,84,84)

                    local_act = episode_local_generator(support_y, query_y, CLASS_NUM, CLASS_SHOT)  # (75,5)
                    fpn_score1, fpn_score2, fpn_score3, scores_0, scores_1, scores_2, scores_3, scores_4 = model(support_x, query_x, CLASS_NUM,
                                                                             CLASS_QUERY, FEATURE_DIM)

                    # 直接or除以3
                    f123 = (fpn_score1 + fpn_score2 + fpn_score3) / 3.0
                    scores_multi = scores_0 + scores_1 + scores_2 + scores_3 + scores_4

                    # 预测指标
                    predict_labels = scores_multi + fpn_score1 + fpn_score2 + fpn_score3
                    predict_labels_f123 = scores_multi + f123

                    _, predict_labels = torch.max(predict_labels.data, 1)
                    _, predict_labels_f123 = torch.max(predict_labels_f123.data, 1)
                    _, local = torch.max(local_act.data, 1)

                    # 4个得分计算
                    rewards = predict_labels.eq(local.to(device)).sum().item()
                    rewards_f123 = predict_labels_f123.eq(local.to(device)).sum().item()

                    # 准确率计算
                    test_accuracy = rewards / 1.0 / (args.way * args.query)
                    accuracies.append(test_accuracy)
                    test_accuracy_f123 = rewards_f123 / 1.0 / (args.way * args.query)
                    accuracies_f123.append(test_accuracy_f123)

                test_accuracy, h = mean_confidence_interval(accuracies)
                test_accuracy_f123, h_f123 = mean_confidence_interval(accuracies_f123)

                xtest.append((train_episode + 1) / 100)
                test_accuracies_list.append(test_accuracy_f123 * 100)

                if test_accuracy_f123 > max_test_accuracy:
                    max_test_accuracy = test_accuracy_f123
                    max_test_accuracy_episode = train_episode + 1

                print("Four indicators:", file=F_txt)
                print("Four indicators:")

                print("test_accuracy:", test_accuracy, "h", h, "max_test_accuracy", max_test_accuracy, "episode",
                      max_test_accuracy_episode, file=F_txt)
                print("test_accuracy:", test_accuracy, "h", h, "max_test_accuracy", max_test_accuracy, "episode",
                      max_test_accuracy_episode)

                print("test_accuracy_f123:", test_accuracy_f123, "h_f123", h_f123)

                print("test_accuracy_f123:", test_accuracy_f123, "h_f123", h_f123, file=F_txt)

                if test_accuracy_f123 > last_accuracy:
                    # save networks
                    torch.save(model.state_dict(),
                               "./models/" + "Episode" + str(train_episode + 1) + "-"
                               + "test_accuracy-" + str(test_accuracy_f123) + ".pkl")
                    print("save networks for episode:", train_episode + 1, file=F_txt)
                    print("save networks for episode:", train_episode + 1)
                    last_accuracy = test_accuracy_f123

                if test_accuracy_f123 > last_accuracy or test_accuracy_f123 > base_accuracy:
                    # save networks
                    torch.save(model.state_dict(),
                               "./models/check_point/" + "Episode" + str(train_episode + 1) + "-"
                               + "test_accuracy-" + str(test_accuracy_f123) + ".pkl")
                    print("save networks for episode:", train_episode + 1, file=F_txt)
                    print("save networks for episode:", train_episode + 1)

        if (train_episode + 1) % 100000 == 0:
            # save networks
            torch.save(model.state_dict(),
                       "./models/" + args.feature_encoder + "-" + str(args.dataset) + "episode" + str(train_episode + 1)
                       + "-" + str(args.way) + "-way-" + str(args.shot) + "-shot.pkl")
            print("save networks for episode:", train_episode + 1, file=F_txt)
            print("save networks for episode:", train_episode + 1)

        # online monitor
        if (train_episode + 1) % 2500 == 0:
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(xtrain, accuracy_list, 'o-', xval, val_accuracies_list, 'r--', xtest, test_accuracies_list, 'm:')
            plt.title("accuracy of train-val" + str(args.episode) + "-episode")
            plt.ylabel("accuracy")
            plt.subplot(2, 1, 2)
            plt.plot(xloss, loss_list, '.-')
            plt.xlabel("loss of train")
            plt.ylabel("train loss")
            plt.pause(0.01)
            plt.ioff()
    F_txt.close()
    plt.show()


def main():
    # Step 1:init data folders
    print("init data folders")
    if args.dataset == "miniImagenet":
        metatrain_folder, metaval_folder, metatest_folder = mini_imagenet_folder(args.train_folder, args.val_folder, args.test_folder)
    else:
        metatrain_folder, metaval_folder, metatest_folder = mini_imagenet_folder(args.train_folder, args.val_folder, args.test_folder)
    task_generator = TaskGenerator(metatrain_folder, metaval_folder, metatest_folder)

    # Step 2:init Model
    print("init MsCRN Neural Network")
    model = MsCRN(args.way, args.shot, args.query, args.feature_encoder).apply(weights_init_kaiming)
    model.to(device)
    # if args.train_embedding:
    #     embedding_train(model, task_generator)
    #     torch.cuda.empty_cache()
    # relation_train(model, task_generator)
    train(model, task_generator)


if __name__ == '__main__':
    main()
