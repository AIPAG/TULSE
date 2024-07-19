
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn import functional as F
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention as SDPAtt

class AttHead(nn.Module):
    def __init__(self, size=4) -> None:
        ''' 一个注意力机制 '''
        super().__init__()
        self.w = torch.randn(1, 1, size, requires_grad=True).cuda()

    def forward(self, x:torch.Tensor, hn:torch.Tensor):
        # hn: batch_size * 1 * hidden_size
        # x: maxlen * batch_size * hidden_size
        t = self.w * x
        t = t.permute(1, 2, 0)
        # t: batch_size * hidden_size * maxlen
        atten_energies = torch.bmm(hn, t)
        scores = F.softmax(atten_energies, dim=2)
        t = t.permute(0, 2, 1)  # 32*len*400
        out = torch.bmm(scores, t)  # 32*1*400       #hidden_size * batch_size
        return out
    
class MultiHead(nn.Module):
    def __init__(self, hidden_size=4, num_heads=2) -> None:
        ''' 多头注意力机制 '''
        super().__init__()
        self.heads = nn.ModuleList([AttHead(hidden_size) for _ in range(num_heads)])

    def forward(self, y:torch.Tensor, hn:torch.Tensor):
        hn = hn.permute(1, 0, 2)
        # hn: batch_size * 1 * hidden_size
        # y: maxlen * batch_size * hidden_size
        # head_outs = torch.stack([head(y, hn) for head in self.heads])
        # ave_out = torch.mean(head_outs, dim=0)
        # out = hn*ave_out
        out = torch.cat([hn*head(y, hn) for head in self.heads], dim=2)
        out = out.permute(1, 0, 2)
        return out

class MHLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, batch_size=1, num_heads=2, bilstm=False) -> None:
        ''' 带多头注意力的LSTM '''
        super().__init__()
        t = 2 if bilstm else 1
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bilstm)
        self.mulhead = MultiHead(hidden_size*t, num_heads)
        # self.att = SDPAtt(d_model=hidden_size*t, d_k=hidden_size*t, d_v=hidden_size, h=num_heads)
        self.h0 = torch.zeros(num_layers*t, batch_size, hidden_size).cuda()
        self.c0 = torch.zeros(num_layers*t, batch_size, hidden_size).cuda()

    def forward(self, x:torch.Tensor, traj_lens):
        # x: maxlen * batch_size * embed_size
        x = rnn_utils.pack_padded_sequence(x, traj_lens)
        y, (hn, _) = self.layer1(x, (self.h0, self.c0))
        # hn: t * batch_size * hidden_size
        y, ylen = rnn_utils.pad_packed_sequence(y)
        hn:torch.Tensor = hn.view(1, hn.size(1), -1)
        # y: maxlen * batch_size * hidden_size
        out = self.mulhead(y, hn)
        # out = torch.cat([self.mulhead(y, hn), hn], dim=2)
        # out = self.att(y, y, y)
        out = torch.mean(out, dim=0, keepdim=True)
        out = torch.cat([out, hn.view(1, out.size(1), -1)], dim=2)
        return out

class MHFCLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, batch_size=1, num_heads=2, bilstm=False) -> None:
        ''' 带多头注意力的LSTM '''
        super().__init__()
        t = 2 if bilstm else 1
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bilstm)
        self.mulhead = MultiHead(hidden_size*t, num_heads)
        self.h0 = torch.zeros(num_layers*t, batch_size, hidden_size).cuda()
        self.c0 = torch.zeros(num_layers*t, batch_size, hidden_size).cuda()

    def forward(self, x:torch.Tensor, traj_lens):
        # x: maxlen * batch_size * embed_size
        x = rnn_utils.pack_padded_sequence(x, traj_lens)
        y, (hn, _) = self.layer1(x, (self.h0, self.c0))
        # hn: t * batch_size * hidden_size
        y, ylen = rnn_utils.pad_packed_sequence(y)
        hn:torch.Tensor = hn.view(1, hn.size(1), -1)
        # y: maxlen * batch_size * (hidden_size*t)
        out = self.mulhead(y, y)
        # out = self.mulhead(y, torch.mean(y, dim=0, keepdim=True))
        out = torch.mean(out, dim=0, keepdim=True)
        out = torch.cat([out, hn], dim=2)
        return out

# v-1.1: 将TULAM的各个组成部分拆分了，更直观，并且也稍微美化了代码
class TULAM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1, num_heads=2, bilstm=False):
        super().__init__()
        self.mhlstm = MHLSTM(input_size, hidden_size, num_layers, batch_size, num_heads, bilstm)
        if bilstm:
            hidden_size <<= 1
        self.linears = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear((num_heads+1)*hidden_size, hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(0.6),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x:torch.Tensor, trlen):
        out = self.mhlstm(x, trlen)
        out = self.linears(out)
        return out

class ECA(nn.Module):
    def __init__(self, in_channel:int, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2)+b)/gamma))
        kernel_size = k if k % 2 else k+1
        padding = kernel_size//2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out*x

class TULAM_CNN(nn.Module):
    def __init__(self, lrow:int, lcol:int, num_class:int) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 10, 5)
        # self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 5, 5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(5, 10, 5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Flatten(), 
        )
        self.eca = ECA(10)
        # lrow = ((lrow-4)//2-4)//2
        # lcol = ((lcol-4)//2-4)//2
        lrow //= 4
        lcol //= 4
        self.wide = lrow*lcol*10
        self.network = nn.Sequential(
            nn.Linear(self.wide, num_class), 
            nn.ReLU(), 
            nn.Linear(num_class, num_class), 
        )

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = self.conv_layer(x)
        # out = self.eca(out)
        # out = out.view(-1, self.wide)
        out = self.network(out)
        return out
    
class VGGblock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, kernel_size=3, padding=1, stride=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        out = self.relu(self.bn(self.conv(x)))
        return out
    
class VGG(nn.Module):
    def __init__(self, lrow:int, lcol:int, num_classes:int) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            VGGblock(1, 8), 
            VGGblock(8, 8), 
            nn.MaxPool2d(2), 
        )
        self.layer2 = nn.Sequential(
            VGGblock(8, 16), 
            VGGblock(16, 16), 
            nn.MaxPool2d(2), 
        )
        self.layer3 = nn.Sequential(
            VGGblock(16, 32), 
            VGGblock(32, 32), 
            VGGblock(32, 32), 
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            VGGblock(32, 64), 
            VGGblock(64, 64), 
            VGGblock(64, 64), 
            nn.MaxPool2d(2)
        )
        self.layer5 = nn.Sequential(
            VGGblock(64, 64), 
            VGGblock(64, 64), 
            VGGblock(64, 64), 
            nn.MaxPool2d(2)
        )
        lrow, lcol = lrow//32, lcol//32
        self.linear1 = nn.Sequential(
            nn.Linear(64*lrow*lcol, 4096), 
            nn.ReLU(), 
            nn.Dropout(), 
        )
        self.linear2 = nn.Sequential(
            nn.Linear(4096, 4096), 
            nn.ReLU(), 
            nn.Dropout(), 
        )
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

class TULAM_COMB(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, num_classes:int, batch_size:int, num_heads:int, lrow:int, lcol:int, alpha=0.5) -> None:
        super().__init__()
        self.tulam = TULAM(input_size, hidden_size, num_layers, hidden_size, batch_size, num_heads)
        # self.cnn = TULAM_CNN(lrow, lcol, hidden_size)
        self.cnn = VGG(lrow, lcol, hidden_size)
        # self.linear = nn.Linear(hidden_size*2, num_classes)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*2, num_classes), 
            nn.ReLU(), 
            nn.Linear(num_classes, num_classes), 
        )
        self.alpha = alpha

    def forward(self, traj:torch.Tensor, trlen:int, graph:torch.Tensor):
        out1:torch.Tensor = self.tulam(traj, trlen)
        out1 = out1.squeeze(0)
        out2 = self.cnn(graph)
        out = torch.cat([out1*self.alpha, out2*(1-self.alpha)], dim=1)
        # out = out1*self.alpha+out2*(1-self.alpha)
        out = self.linear(out)
        return out

class TULAM_COMB_1(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, num_classes:int, batch_size:int, num_heads:int, lrow:int, lcol:int, alpha=0.5) -> None:
        super().__init__()
        self.mhlstm = MHLSTM(input_size, hidden_size, num_layers, batch_size, num_heads)
        self.linears = nn.Sequential(
            nn.Dropout(0.6), 
            nn.Linear((num_heads+1)*hidden_size, hidden_size), 
            nn.ReLU(), 
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 5, 5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(5, 10, 5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )
        self.eca = ECA(10)
        lrow //= 4
        lcol //= 4
        self.wide = lrow*lcol*10
        self.network = nn.Sequential(
            nn.Linear(self.wide, hidden_size), 
            nn.ReLU(), 
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*2, num_classes), 
            nn.ReLU(), 
            nn.Linear(num_classes, num_classes), 
        )
        self.alpha = alpha

    def forward(self, traj:torch.Tensor, trlen:int, graph:torch.Tensor):
        out1:torch.Tensor = self.mhlstm(traj, trlen)
        out1 = self.linears(out1)
        out1 = out1.squeeze(0)
        out2:torch.Tensor = self.conv_layer(graph)
        out2 = self.eca(out2)
        out2 = out2.view(-1, self.wide)
        out2 = self.network(out2)
        out = torch.cat([out1*self.alpha, out2*(1-self.alpha)], dim=1)
        # out = out1*self.alpha+out2*(1-self.alpha)
        out = self.linear(out)
        return out

class TextCNN_convblock(nn.Module):
    def __init__(self, embed_size:int, out_channel:int, conv_size:int, seq_size:int):
        super().__init__() # input: batch_size * 1 * seq_size * embed_size
        self.conv = nn.Conv2d(1, out_channel, (conv_size, embed_size)) # batch_size * out_channel * (seq_size-conv_size+1) * 1
        self.pool = nn.MaxPool2d((seq_size-conv_size+1, 1)) # batch_size * out_channel * 1 * 1

    def forward(self, x):
        out:torch.Tensor = self.conv(x)
        out = self.pool(out)
        out = out.view(x.size(0), -1)
        return out

class TextCNN(nn.Module):
    def __init__(self, embed_size:int, out_channel:int, seq_size:int, filter_sizes:list, num_classes:int, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([TextCNN_convblock(embed_size, out_channel, size, seq_size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(filter_sizes)*out_channel, num_classes)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.dropout(out)
        out = self.linear(out)
        return out