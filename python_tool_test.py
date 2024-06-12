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


def left_foot_into_company(days):
    """
    左脚踏入公司被开除
    """
    if days < 1:
        raise Exception("入职天数不合法")
    total_money = 0
    for i in range(1, days+1):
        cur_day_money = 2 ** (i - 1)
        print("第%d天工资为：%d元" % (i, cur_day_money))
        total_money = total_money + cur_day_money
    print("总工资为：%d元" % total_money)


if __name__ == "__main__":
    # print(get_random_number(0, 1))
    # print(get_now_time())
    # print(deng_cha())
    left_foot_into_company(0)
