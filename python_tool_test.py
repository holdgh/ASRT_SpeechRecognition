import random
import time
import numpy as np


def get_random_number(down_bound, up_bound):
    """
    随机输出区间[down_bound, up_bound]内的整数
    注意是闭区间，包含两端
    """
    return random.randint(down_bound, up_bound)


def get_now_time():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def deng_cha():
    """
    等差数列 第一个参数为起始值，第二个参数为终止值，第三个参数为元素数量【根据这三个参数得出等差数列】，第四个参数为数据类型
    """
    return np.linspace(0, 400 - 1, 10, dtype=np.int64)


if __name__ == "__main__":
    # print(get_random_number(0, 1))
    # print(get_now_time())
    print(deng_cha())
