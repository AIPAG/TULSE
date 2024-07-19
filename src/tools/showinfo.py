import pandas as pd

from time import time

class mytimer:
    def __init__(self, desc:str="No Name timer") -> None:
        ''' 计时器类。实现了计时器，并且在出现exception时会正常报错但仍会显示显示总耗时 '''
        self.desc = desc

    def __repr__(self) -> str:
        return f"Current timer: {self.desc}"
    
    def __enter__(self):
        self.st = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"exception type: {exc_type}")
            print(f"exception value: {exc_val}")
            print(f"exception tb: {exc_tb}")
        self.et = time()
        print(f"{self.desc}: total time cost is {self.et-self.st:.2f}s.")

    def show_time(self) -> None:
        tt = time()
        print(f"{self.desc}: time cost til now is {tt-self.st:.2f}s.")

class const_timer(mytimer):
    def __enter__(self):
        self.start()
        return self

    def start(self) -> None:
        self.lt = time()
        self.st = self.lt

    def record(self, rec:bool=True) -> float:
        t = time()
        ret = t-self.lt
        if rec:
            self.lt = t
        return ret
    
    def record_til_start(self, rec:bool=True) -> float:
        t = time()
        if rec:
            self.lt = t
        return t-self.st


# 之所以要使用修饰器主要是因为：
# 一方面增减计时器都只需要一行代码。
# 另一方面不会影响编译器在原函数上显示的信息，也不用为每个需要计时器的函数都单独写一个带计时器的函数
def func_with_timer(desc:str="No Name timer"):
    ''' 修饰器。在函数前加上@func_with_timer(desc)即可实现计时，其中desc为计时器名称 '''
    def run_time(func):
        def wrap(*args, **kwargs):
            with mytimer(desc=desc):
                ret = func(*args, **kwargs)
            return ret
        return wrap
    return run_time

def show_dataset_info(descs=[]):
    '''  显示返回数据集的信息。对于每一个DataFrame返回值，需要输入一个desc，并请确保数据集有userID、TrID这2列 '''
    def show_info(func):
        def show(*args, **kwargs):
            res = func(*args, **kwargs)
            i = 0
            l = len(descs)
            for output in res:
                if type(output)!=pd.DataFrame:
                    continue
                uids = output["userID"].drop_duplicates()
                trajs = output[['userID', 'TrID']].drop_duplicates()
                if i<l:
                    desc = descs[i]
                else:
                    desc = "dataset info"
                print(f"{desc}: \n\tnum of user: {len(uids)}\n\tnum of trajs: {len(trajs)}\n\tnum of checkins: {len(output)}\n\taverage traj length: {len(output)/len(trajs):.1f}")
                i += 1
            return res
        return show
    return show_info