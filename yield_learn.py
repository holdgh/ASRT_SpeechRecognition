def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res: ", res)


def use_yield():
    """
    带yield的函数是一个生成器，而不是一个函数了，这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数，这一次的next开始的地方是接着上一次的next停止的地方执行的，所以调用next
    的时候，生成器并不会从foo函数的开始执行，只是接着上一步停止的地方开始，然后遇到yield后，return出要生成的数，此步就结束。
    为什么要使用生成器？如果使用集合的数据结构，将会占用很多内存空间。使用生成器可以节省空间，使用next方法即可访问数据。比如遍历打印数字1到100
    """
    g = foo()  # 将生成器赋值为g，此时“函数foo”并未执行打印功能
    print(next(g))  # next获取生成器的一个元素【可否理解返回一个函数体？】【从“函数foo”第一行到yield修饰的语句【return其后的数据】【注意并未包含赋值操作】】
    print("*" * 20)
    print(next(
        g))  # 再次next，获取生成器的一个元素【从上一次next停止未执行的地方【对于res赋值操作，注意上次next已经将4返回了，此时赋值的右边没有值了，为None】通过while语句体到yield
    # 修饰的语句【return关键字yield后的数据】】
    print("*" * 20)
    print(next(
        g))  # 再次next，获取生成器的一个元素【从上一次next停止未执行的地方【对于res赋值操作，注意上次next已经将4返回了，此时赋值的右边没有值了，为None】通过while语句体到yield
    # 修饰的语句【return关键字yield后的数据】】
    print("*" * 20)
    # 生成器的send方法注意事项：不能在首次使用生成器时传递非None数值，否则抛出异常TypeError。也即首次使用迭代器且调用send函数时必须传None
    print(g.send(16))  # 生成器的send函数作用：给上次next终止的地方传值+再次执行next


def print_1_to_100_by_list():
    """
    使用range函数打印数字
    """
    for i in range(100):
        print(i + 1)


def generator_list(bound):
    """
    定义一个生成器，用以从小到大获取边界值及以下的数字
    """
    i = 0
    while i < bound:
        i = i + 1
        yield i


def print_1_to_100_by_generator():
    for i in generator_list(100):
        print(i)


if __name__ == "__main__":
    # use_yield()
    # print_1_to_100_by_list()
    print_1_to_100_by_generator()
