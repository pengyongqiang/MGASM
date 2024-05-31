from align_utils import get_label


class User:
    # 构造函数，在创建对象时调用
    def __init__(self, row, dataset):
        # 声明类的属性
        self.char = row[0]
        self.word = row[1]
        self.article = row[2]
        if dataset == 'wd':
            self.label = get_label(row[2])
        else:
            self.label = row[3]
