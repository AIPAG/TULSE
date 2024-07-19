import argparse
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import warnings

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.project.model import TULAM
from src.tools.encoder import timeslicing, visitMatrix, addLIDtoUserSubT, dataset_w2v
from src.tools.get_args_logger import get_args_logger
from src.tools.preprocess import splitData, toTenser, coordinate2grid, checkins2subtraj, filt_trajnum
from src.tools.loaddataset import loadDataset
from src.tools.test_evaluate import evaluate_model

warnings.filterwarnings("ignore")

'''
将i/o与训练模型以外的代码全部挪到别的文件夹里面去了，现在main文件简洁多了
'''

class TrData(Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]

# v-1.2: 调整了返回值的顺序，以增加可扩展性。返回值应该为*args, y的形式，其中args为模型的输入，y为预测标签
def collate_fn(trs:list[np.array]):
    trs.sort(key=lambda x: len(x), reverse=True)
    tr_lengths = [len(sq) for sq in trs] # 创建一个tr_lengths列表，每个值都是轨迹的真实长度（有多少位置点）
    trs = rnn_utils.pad_sequence(trs, padding_value=0) # 从数据集中取出时，自动填充。通用的

    var_x = trs[:, :, 1:] # maxlen * batch_size * word_size
    var_y = trs[0, :, 0]

    return var_x, tr_lengths, var_y


def train_model(model, train_data, batch_size, num_epochs, learning_rate, test_data, num_classes):
    train_data_inBatch = TrData(train_data) # 采用DataLoader从数据集中获取数据（Batch方式）
    # 一个Batch取5条轨迹，在取值的时候是会shuffle的。包括x和y，但在trainloader取出时会由collate_fn拆分
    train_loader = DataLoader(train_data_inBatch, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

    # 设置优化方法和loss
    criterion = nn.CrossEntropyLoss()
    # nn.CrossEntropyLoss里面包含了softmax层，
    # 所以我们在使用nn.CrossEntropyLoss来计算loss的时候，
    # 不需要再加softmax层作为输出层，直接使用最后输出的特征向量来计算loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 增加自动调整学习率的部分，每10次调用减少为原来的0.8倍
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    losslist = []
    acclist = []
    top_quant_acc_list, bottom_quant_acc_list = [], []
    # model.train()#测试时，根本就没有反向传播调整参数的代码，因此不用担心测试的时候仍然会训练
    # model.train和eval仅仅用于使用了BN和Dropout时，训练集有一个dropout rate=0.5，
    # 而测试时要求dropout rate=1，所以通过train和eval两个函数通知框架，模型是处于什么状态，而选什么参数值。
    # 至于测试时，梯度是否在计算和累加，答案是如果不在测试阶段显式加入no_grad去掉梯度。因为测试阶段不需要反向传播梯度，这个计算是没用的。
    # 但是因为测试的代码里你没有写反向传播的代码，所以这个梯度的计算不会对模型产生影响，只是费一些内存而已。
    # 所以测试阶段加不加no_grad去掉梯度都可以。
    for epoch in tqdm(range(num_epochs), desc='Training...'):
        model.train()
        ave_loss_of_epoch = 0
        count = 0
        for step, (X_vector, len_of_oneTr, Y_vector) in enumerate(train_loader):
            # 注意！cuda()是返回一个在cuda上的复制，不会更改原tensor
            X_vector = X_vector.cuda()# 590*64*102
            Y_vector = Y_vector.cuda()
            out:torch.Tensor = model(X_vector, len_of_oneTr)

            # 把out从[1，5，182]压缩为[5,182]
            out = out.squeeze(0)
            # cross_entropy target参数只需要标签即可, 不需要传one-hot向量，且数值必须是long，不能是float
            Y_vector:torch.Tensor = Y_vector.long()
            loss = criterion(out, Y_vector)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 用于统计
            ave_loss_of_epoch = ave_loss_of_epoch + loss
            count = count + 1
            time.sleep(1/32) # Gowalla-D
            # time.sleep(1/8+1/4) # Geolife

        # 更新学习率lr，放在epoch这一层循环中。
        if optimizer.param_groups[0]['lr'] > 2e-5:
            scheduler.step()

        ave_loss_of_epoch = ave_loss_of_epoch / count
        losslist.append(ave_loss_of_epoch.item())
        # 可视化代码
        test_loader = DataLoader(TrData(test_data), batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True)
        *_, top1_correct_Rate, (top_quant_acc, _, _, bottom_quant_acc) = evaluate_model(model, test_loader, num_classes, 4, (1,))

        acclist.append(top1_correct_Rate)
        top_quant_acc_list.append(top_quant_acc)
        bottom_quant_acc_list.append(bottom_quant_acc)

    return model, losslist, acclist, top_quant_acc_list, bottom_quant_acc_list

def main():
    print('TUL checker...')
    args, logger = get_args_logger()
    cachePath = './tmp/'
    logger.info('configure:')
    logger.info(f'\tdataSet: {args.dataset}')
    logger.info(f'\tbatchSize: {args.batchSize}')
    logger.info(f'\tepochs: {args.epochs}')
    logger.info(f'\tnumHeads: {args.numHeads}')

    model_name = "TULSE"
    dataset_name = f"{args.dataset}_{'D' if args.isDense else 'S'}"
    model_path = f"./model/{model_name}_{dataset_name}.pth"
    filename = f"{dataset_name}.csv"
    train_path = f"./dataset/{dataset_name}_train.csv"
    test_path = f"./dataset/{dataset_name}_test.csv"

    st = time.time()
    # load checkin data
    if args.load_train_test:
        trainSet = pd.read_csv(train_path)
        testSet = pd.read_csv(test_path)
        numTrain = len(trainSet[['userID', 'TrID']].drop_duplicates())
        numTest = len(testSet[['userID', 'TrID']].drop_duplicates())
    elif args.loadDataset:
        trajectories = pd.read_csv(cachePath+filename)
        addLIDtoUserSubT(trajectories)
        trainSet, testSet, numTrain, numTest = splitData(trajectories, args.testNum)
        trainSet.to_csv(train_path, mode='w', index=False, header=True, encoding='utf_8_sig')
        testSet.to_csv(test_path, mode='w', index=False, header=True, encoding='utf_8_sig')
    else:
        allCheckin, checkinNum, userNum = loadDataset(args)
        logger.info(f'num of checkin: {checkinNum}\nnum of user: {userNum}')

        # trans latitude and longitude to grid ID
        coordinate2grid(allCheckin, args.gridSize)

        # trans checkin into trajectory
        trajectories, trNum = checkins2subtraj(allCheckin, args.intervalHour, 60)
        trajectories, trNum = filt_trajnum(trajectories, args.trajThreshold)
        logger.info('num of trajectory: {}'.format(trNum))
        if not os.path.exists(cachePath):
            os.makedirs(cachePath)
        trajectories.to_csv(cachePath + filename, mode='w', index=False, header=True, encoding='utf_8_sig')

        addLIDtoUserSubT(trajectories)
        # split train-test dataset
        trainSet, testSet, numTrain, numTest = splitData(trajectories, args.testNum)
        trainSet.to_csv(train_path, mode='w', index=False, header=True, encoding='utf_8_sig')
        testSet.to_csv(test_path, mode='w', index=False, header=True, encoding='utf_8_sig')
    logger.info(f'num of trajectory in trainSet: {numTrain}')
    logger.info(f'num of trajectory in testSet: {numTest}')

    # embedding---------------------------------------------------------------------------------------------------
    # deepwalk
    '''
    deepwalk = deepwalkEmbedding(trajectories, trainSet, testSet, args.embedSize)
    trainData = pd.merge(trainSet, deepwalk, on=['lid'])
    testData = pd.merge(testSet, deepwalk, on=['lid'])
    
    # approximateOnehot
    trainData, testData, onehotSize = approximateOnehotEmbed(trajectories, trainSet, testSet)
    '''
    # RE
    # count timeslice and visit matrix
    visit_train = visitMatrix(trainSet)
    # visit_train.to_csv(f"{cachePath}visit_train.csv")
    # timeslice_train = timeslicing(trainSet)
    # timeslice_train.to_csv(f"{cachePath}timeslice.csv")
    # w2v_train = dataset_w2v(trainSet, args.embedSize)
    trainData = pd.merge(trainSet, visit_train, on='lid', how='left')
    # trainData = pd.merge(trainData, timeslice_train, on='lid', how='left')
    # trainData = pd.merge(trainData, w2v_train, on='lid', how='left')

    maptable = trainData.drop(columns=['userID','TrID','rowID','colID','utc'])
    maptable.drop_duplicates(inplace=True)
    testData = pd.merge(testSet, maptable, on='lid', how='left', indicator=True)
    testData['filter'] = testData['_merge']=='both'
    testData = testData[testData['filter']==True]
    testData.drop(['filter', '_merge'], axis=1, inplace=True)

    # RE done
    logger.info(f'embedding done. size of one point: {trainData.shape[1]}')
    # embedding done---------------------------------------------------------------------------------------------------
    trainData = toTenser(trainData)
    testData = toTenser(testData)

    et = time.time()
    print(f"preprocess time is {et-st:.2f} s")

    # build model
    oneTr = trainData[0]
    onePointSize = oneTr.size(1)
    inputSize = onePointSize - 1
    num_classes = max(trainSet['userID'].unique()) + 1

    model = TULAM(input_size=inputSize,
                      hidden_size=args.hiddenSize,
                      num_layers=1,
                      num_classes=num_classes,
                      batch_size=args.batchSize,
                      num_heads=args.numHeads, 
                      bilstm=False)
    model.cuda()

    # train
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    mymodel, losslist, acclist, tqacc, bqacc = train_model(model, trainData, args.batchSize, args.epochs, args.lr, testData, num_classes)
    end_time.record()
    torch.cuda.synchronize()
    logger.info("train done.time cost: {}".format(start_time.elapsed_time(end_time)))
    torch.save(model, model_path)

    test_loader = DataLoader(TrData(testData), batch_size=args.batchSize, shuffle=True,
                             collate_fn=collate_fn, drop_last=True)
    macroP, macroR, macroF1, top1acc, top3acc, top5acc, top10acc, acc_qt = evaluate_model(mymodel, test_loader, num_classes, args.quantile, (1, 3, 5, 10))

    logger.info(f"Macro-P: {macroP}")
    logger.info(f"Macro-R: {macroR}")
    logger.info(f"Macro-F1: {macroF1}")

    # 输出topk准确率
    logger.info(f'Test top 1 Accuracy of the model on the testdata: {top1acc}')
    logger.info(f'Test top 3 Accuracy of the model on the testdata: {top3acc}')
    logger.info(f'Test top 5 Accuracy of the model on the testdata: {top5acc}')
    logger.info(f'Test top 10 Accuracy of the model on the testdata: {top10acc}')
    logger.info(f'Test accuracy of top {args.quantile} quantile: {acc_qt[0]}')
    print(acc_qt)
    
    try:
        with open('./log/record.csv', 'a', newline='') as frec:
            rec_writer = csv.writer(frec)
            # rec_writer.writerow([])
            rec_writer.writerow([args.dataset, model_name, macroP, macroR, macroF1, top1acc, top3acc, top5acc, top10acc, args.gridSize, args.isDense] + acc_qt)
    except PermissionError as pe:
        print(pe)

    plt.title("traning accuracy trend")
    plt.plot([i for i in range(len(acclist))], acclist, linewidth=0.5)
    # plt.plot([i for i in range(len(tqacc))], tqacc, linewidth=0.5)
    # plt.plot([i for i in range(len(bqacc))], bqacc, linewidth=0.5)
    plt.xlabel("epoch")
    plt.ylabel("accuracy on test dataset")
    plt.show()

    logger.info('Done')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
