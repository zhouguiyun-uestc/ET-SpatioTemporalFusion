import torch
from torch import nn
import math

EPS = 1e-8

class ETModel(nn.Module):
    #time_steps:输入的时间长度
    #num_spatial：空间上的卷积层个数
    #num_sequential：时间上的GRU个数
    #look_ahead：预测的时间长度-输入的时间长度
    def __init__(self, feature_dim, time_steps, hidden_dim, num_spatial, num_sequential, look_ahead):
        super(ETModel, self).__init__()
        self.feature_dim = feature_dim
        self.time_steps = time_steps + look_ahead # 输出的时间长度
        self.look_ahead = look_ahead

        #Attention
        #self attention 标准化代码
        self.linear_k = nn.Linear(feature_dim, feature_dim)
        self.linear_q = nn.Linear(feature_dim, feature_dim)
        self.linear_v = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.gn = GlobalLayerNorm(feature_dim, self.time_steps)

        in_dims = [2 ** (i+5) if i > 0 else feature_dim for i in range(num_spatial)]
        out_dims = [2 ** (i+6) for i in range(num_spatial)]
        spatial_convs = []
        spatial_deconvs = []
        for i in range(num_spatial):
            spatial_convs += [ConvBlock(in_dims[i], out_dims[i], \
                                        sequential_dim=self.time_steps, kernel_size=3, \
                                        stride=2, dilation=1, padding=1, activation="PReLU")]


            act = "PReLU" if i < num_spatial-1 else "None"
            in_dim = in_dims[num_spatial-i-1] if i < num_spatial-1 else feature_dim+1
            shortcut = True if i < num_spatial-1 else False
            spatial_deconvs += [ConvTransposeBlock(out_dims[num_spatial-i-1], in_dim, \
                                                    kernel_size=3, sequential_dim=self.time_steps,\
                                                    stride=2, dilation=1, \
                                                    padding=1, shortcut=shortcut, \
                                                    activation=act)]
        
        self.spatial_convs = nn.ModuleList(spatial_convs)
        self.spatial_deconvs = nn.ModuleList(spatial_deconvs)
        self.sequential = SequentialBlock(out_dims[-1], out_dims[-1], \
                                          hidden_dim, self.time_steps, num_sequential)

        in_dims = [2 ** (i+3) if i > 0 else feature_dim+1 for i in range(num_spatial)]
        out_dims = [2 ** (i+4) for i in range(num_spatial)]
        dilations = [2 ** i for i in range(num_spatial)]
        convs = []
        deconvs = []
        for i in range(num_spatial):
            convs += [ConvBlock(in_dims[i], out_dims[i], \
                        sequential_dim=self.time_steps, kernel_size=3, \
                        stride=1, dilation=dilations[i], padding=2*dilations[i], activation="PReLU")]

            act = "PReLU" if i < num_spatial-1 else "None"
            in_dim = in_dims[num_spatial - i - 1]
            deconvs += [ConvTransposeBlock(out_dims[num_spatial - i -1], in_dim, \
                                            kernel_size=3, sequential_dim=self.time_steps,\
                                            stride=1, dilation=dilations[num_spatial-i-1], \
                                            padding=2*dilations[num_spatial-i-1], shortcut=shortcut, \
                                            activation=act)]
        self.convs = nn.ModuleList(convs)
        self.deconvs = nn.ModuleList(deconvs)
    
    def forward(self, x):
        #B:batch_size, C:number of feature, W:width, H:height, T:time
        B, C, W, H, T = x.shape
        x = torch.cat([x, torch.zeros((B, C, W, H, self.look_ahead), \
                            dtype=x.dtype, device=x.device)], dim=-1)

        x = x.permute(0,2,3,4,1).reshape(-1, C)

        #Attention
        k,q,v = self.linear_k(x), self.linear_q(x), self.linear_v(x)
        w = torch.matmul(k.unsqueeze(dim=-1), q.unsqueeze(dim=1)) / math.sqrt(C)
        w = self.softmax(w)
        x = torch.matmul(w, v.unsqueeze(dim=-1)).squeeze(dim=-1)
        x = self.relu(self.linear(x))
        x = self.gn(x.reshape(B, W, H, -1, C).permute(0, 4, 1, 2, 3))

        residuals = [x]
        for m in self.spatial_convs:
            x = m(x)
            residuals += [x]

        #提取时序信息
        x = self.sequential(x)

        index = len(residuals) - 2
        for m in self.spatial_deconvs:
            x = m(x, residuals[index])
            index -= 1

        residuals = [x]
        for m in self.convs:
            x = m(x)
            residuals += [x]
        index = len(residuals) - 2
        for m in self.deconvs:
            x = m(x, residuals[index])
            index -= 1
        return x.squeeze()


#全局归一化：对输入的数据进行归一化
class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_dim, sequential_dim):
        super(GlobalLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channel_dim, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channel_dim, 1, 1, 1))
    
    def forward(self, x):
        mean = torch.mean(x, (1,2,3,4), keepdim=True)
        var = torch.mean((x - mean)**2, (1,2,3,4), keepdim=True)
        x = (x - mean) / torch.sqrt(var + EPS) * self.weight + self.bias
        return x


class ConvBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 sequential_dim, 
                 kernel_size=3, 
                 stride=1, 
                 dilation=1, 
                 padding=0, 
                 activation="PReLU"):
        super(ConvBlock, self).__init__()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "PReLU":
            self.activation = nn.PReLU()
        else:
            raise NotImplementedError(f"NotImplement {activation}")
        
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, \
                              stride=stride, padding=padding, dilation=dilation)
        self.conv_trans = nn.Conv2d(input_dim, output_dim, kernel_size, \
                                    stride=stride, padding=padding, dilation=dilation)
        self.conv_gated = nn.Conv2d(2 * output_dim, output_dim, 1, \
                                    stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.normalization = GlobalLayerNorm(output_dim, sequential_dim)
    
    def forward(self, x):
        #[B, C, W, H, T] -> [B*T, C, W, H]
        B, C, W, H, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(-1, C, W, H)
        o = self.activation(self.conv(x))
        x = self.conv_trans(x) #not activation
        gate = self.sigmoid(self.conv_gated(torch.cat([x, o], dim=1)))
        o = o * gate + x * (1.0 - gate)
        #[B*T, C, W, H] -> [B, C, W, H, T]
        _, C, W, H = o.shape
        o = o.reshape(B, -1, C, W, H).permute(0, 2, 3, 4, 1)
        o = self.normalization(o)
        return o


class ConvTransposeBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 sequential_dim, 
                 kernel_size=3, 
                 stride=1, 
                 dilation=1, 
                 padding=0, 
                 shortcut=True,
                 activation="PReLU"):
        super(ConvTransposeBlock, self).__init__()
        self.shortcut = shortcut
        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "PReLU":
            self.activation = nn.PReLU()
        elif activation == "None":
            self.activation = None
        else:
            raise NotImplementedError(f"NotImplement {activation}")
        
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, \
                              stride=stride, padding=padding, dilation=dilation)
        self.conv_gated = nn.Conv2d(2 * output_dim, output_dim, 1, \
                                    stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.normalization = GlobalLayerNorm(output_dim, sequential_dim)
    
    def forward(self, x, res):
        #[B, C, W, H, T] -> [B*T, C, W, H]
        B, C, W, H, T = x.shape
        _, Cr, Wr, Hr, _ = res.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(-1, C, W, H)
        res = res.permute(0, 4, 1, 2, 3).reshape(-1, Cr, Wr, Hr)
        o = self.conv(x)
        if self.activation is not None:
            o = self.activation(o)

        _, C, W, H = o.shape
        if W < Wr or H < Hr:
            o = torch.nn.functional.pad(o, (0, Hr-H, 0, Wr-W))
        else:
            o = o[:, :, :Wr, :Hr]
        if self.shortcut: #快捷连接 改进版（通过门控结构）
            mask = self.conv_gated(torch.cat([res, o], dim=1))
            o = mask * o + (1.0 - mask) * res

        #[B*T, C, W, H] -> [B, C, W, H, T]
        o = o.reshape(B, -1, C, Wr, Hr).permute(0, 2, 3, 4, 1)
        if self.shortcut:
            o = self.normalization(o)
        return o


class SequentialBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim, 
                 sequential_dim, 
                 num_layers):
        super(SequentialBlock, self).__init__()
        self.sequential_dim = sequential_dim
        self.sequence_model = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, \
                                     num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.normalization = GlobalLayerNorm(output_dim, sequential_dim)
    
    def forward(self, x):
        #[B, C, W, H, T] -> [B*W*H, T, C]
        B, C, W, H, T = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, T, C)
        o, _ = self.sequence_model(x) #GRU有两个输出，一个是最终输出，一个是每一层的状态。
        o = self.fc(o)
        #[B*W*H, T, C] -> [B, C, W, H, T]
        o = o.reshape(B, W, H, T, -1).permute(0, 4, 1, 2, 3)
        o = self.normalization(o)
        return o


def print_params(model):
    sm = 0.0
    for name, params in model.named_parameters():
        sm += params.numel() #参数量


if __name__ == '__main__':
    model = ETModel(feature_dim = 7,
                     time_steps = 12,
                     hidden_dim = 512, 
                     num_spatial = 3,
                     num_sequential = 2,
                     look_ahead = 1)
    x = torch.rand([1, 7, 90, 180, 12])
    pred = model(x)

    print_params(model)

