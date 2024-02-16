import random
import torch


def local_generator(y, num_class, class_shot):  # (01234) (75)

    nums = len(y)  # b
    support_y = y[:, :num_class * class_shot]  # (b, 5)
    query_y = y[:, -1]  # (b, 1)

    query_y = query_y.unsqueeze(1).repeat(1, num_class)  # (75,5)

    locals_act = support_y.eq_(query_y)

    return locals_act  # (order_nums, class_nums*class_query, class_num)


# def episode_local_generator(sy, qy, num_class, class_shot):  # (01234) (75)
#
#     nums = len(qy)  # b
#     support_y = sy.unsqueeze(0).repeat(nums, 1)  # (b, 5)
#     query_y = qy.unsqueeze(1).repeat(1, num_class)  # (b, 1)
#
#     locals_act = support_y.eq_(query_y)
#
#     return locals_act  # (order_nums, class_nums*class_query, class_num)

def episode_local_generator(sy, qy, num_class, class_shot):  # (01234) (75)

    nums = len(qy)  # b
    sy = sy[0:num_class * class_shot:class_shot]
    support_y = sy.unsqueeze(0).repeat(nums, 1)  # (b, 5)
    query_y = qy.unsqueeze(1).repeat(1, num_class)  # (b, 1)

    locals_act = support_y.eq_(query_y)

    return locals_act  # (order_nums, class_nums*class_query, class_num)
























