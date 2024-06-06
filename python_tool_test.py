import random
import time


def get_random_number(down_bound, up_bound):
    """
    随机输出区间[down_bound, up_bound]内的整数
    注意是闭区间，包含两端
    """
    return random.randint(down_bound, up_bound)


def get_now_time():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


if __name__ == "__main__":
    # print(get_random_number(0, 1))
    print(get_now_time())
