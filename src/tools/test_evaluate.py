import numpy as np
import torch

from sklearn.metrics import precision_score, recall_score, top_k_accuracy_score
from torch.nn import functional as F

def calc_prf(conf_matrix:np.ndarray) -> tuple[float,float,float]:
    ''' 根据混淆矩阵，计算macroP与macroR，最后计算macro-F1 '''
    acc_array = conf_matrix.diagonal()
    # 计算准确率，可能需要填充nan为0
    pre_array = acc_array/conf_matrix.sum(axis=1)
    pre_array[np.isnan(pre_array)] = 0
    macroP = np.mean(pre_array)
    # 计算召回率，理论上来说不会出现nan，出了说明有bug，需要回头调试
    macroR = np.mean(acc_array/conf_matrix.sum(axis=0))
    # 根据其他论文来看，macro-F1计算方法为直接计算macroP与macroR的调和平均，因此这里也采取这种方案
    macroF1 = 2*macroP*macroR/(macroP+macroR)
    return macroP, macroR, macroF1

# v-1.1: 将原来完全没必要的DataFrame修改为数组来加速代码，同时添加了对任意个top k准确率的计算
def evaluate_model(model:torch.nn.Module, test_loader, class_num:int, quantile:int, topk=(1,), savepath=''):
    ''' 根据给定的模型与测试集、类别数和需要的top k准确率，返回指标 '''
    model.eval()
    conf_matrix = [[0]*class_num for _ in range(class_num)]
    counter = 0
    acc_list = [0.]*len(topk)
    mk = max(topk) # 每次直接取top mk个预测值，相当于预处理
    ytrue, yfalse = [0]*class_num, [0]*class_num

    for test_data in test_loader:
        *args, yvec = test_data
        batch_size = yvec.size(0)
        counter += batch_size
        yvec:torch.Tensor = yvec.int()

        # 运行模型，得到预测值
        for i in range(len(args)):
            if type(args[i])==torch.Tensor:
                args[i] = args[i].cuda()
        pred:torch.Tensor = model(*args)
        pred = pred.squeeze(0)
        pred = F.softmax(pred, dim=1)
        pred = pred.detach().cpu()

        # 预处理top mk个预测值
        _, predk = pred.topk(k=mk, dim=1)
        predk:torch.Tensor = predk.t()
        correct = predk.eq(yvec.view(1, -1).expand_as(predk))

        # 对于每个top k再单独从top mk中取，代码比较简洁
        for i, k in enumerate(topk):
            acc_list[i] += float(correct[:k].reshape(-1).float().sum(0, keepdim=True))

        # 借助top 1的结果更新混淆矩阵
        acc1 = predk[0].int()
        # for i in range(batch_size):
        #     conf_matrix[acc1[i]][yvec[i]] += 1
        for i, j in zip(acc1, yvec):
            conf_matrix[i][j] += 1

        # 更新每个用户的正确率
        for i, j in zip(acc1, yvec):
            if i==j:
                ytrue[j] += 1
            else:
                yfalse[j] += 1

    # 计算最终的top k准确率
    for i in range(len(topk)):
        acc_list[i] /= counter

    # 计算每个用户的正确率
    acc_user = [t/(t+f) for t, f in zip(ytrue, yfalse)]
    acc_user.sort(reverse=True)
    acc_quantiles = np.array_split(acc_user, quantile)
    for i in range(quantile):
        acc_quantiles[i] = np.mean(acc_quantiles[i])

    return *calc_prf(np.array(conf_matrix)), *acc_list, acc_quantiles

def evaluate_labal(x:np.array, y:np.array, topk=(1,)):
    acc_list = [0.]*len(topk)
    for i, k in enumerate(topk):
        acc_list[i] = top_k_accuracy_score(y, x, k=k)
    xmax = np.argmax(x, axis=1)
    macroP:float = precision_score(y, xmax, average='macro')
    macroR:float = recall_score(y, xmax, average='macro')
    macroF1 = 2*macroP*macroR/(macroP+macroR)
    return macroP, macroR, macroF1, *acc_list