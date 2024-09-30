"""
Created on Wed Aug 21 18:43:02 2024

@author: mateo-drr
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def conv(ni, nf, ks=3, stride=1, padding=1, padding_mode='reflect', **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                      padding=padding, padding_mode='reflect', **kwargs)
    nn.init.kaiming_normal_(_conv.weight, mode='fan_out')
    return _conv


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class MoDEEncoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(
            num_experts, num_tasks, in_chan, out_chan)
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv2d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )

    def forward(self, x, t):
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(
            num_experts, num_tasks, in_chan, out_chan)

    def forward(self, x, x_skip, t):
        x = self.convt(x)
        x_cat = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv(num_experts, num_tasks, n_in,
                              n_out, kernel_size=5, padding='same')
        self.conv2 = MoDEConv(num_experts, num_tasks, n_out,
                              n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal'):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                nn.InstanceNorm2d(out_chan, affine=True),
                torch.nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(
            num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode='fan_out')
        return weight

    def gen_avgpool_kernel(self, K):
        weight = torch.ones(K, K).mul(1.0 / K ** 2)
        return weight

    def trans_kernel(self, kernel, target_size):
        Hp = (target_size - kernel.shape[2]) // 2
        Wp = (target_size - kernel.shape[3]) // 2
        return F.pad(kernel, [Wp, Wp, Hp, Hp])

    def routing(self, g, N):
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(
            self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(
            self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum('oihw,hw->oihw', self.expert_avg3x3_conv,
                         self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(
            'oihw,hw->oihw', self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights = []
        for n in range(N):
            weight_nth_sample = (
                torch.einsum('oihw,o->oihw', expert_conv5x5, g[n, 0, :])
                + torch.einsum('oihw,o->oihw', expert_conv3x3, g[n, 1, :])
                + torch.einsum('oihw,o->oihw', expert_conv1x1, g[n, 2, :])
                + torch.einsum('oihw,o->oihw', expert_avg3x3, g[n, 3, :])
                + torch.einsum('oihw,o->oihw', expert_avg5x5, g[n, 4, :])
            )
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):
        N = x.shape[0]  # batch size

        g = self.gate(t)
        g = g.view((N, self.num_experts, self.out_chan))
        g = self.softmax(g)

        w = self.routing(g, N)  # mix expert kernels

        if self.training:
            y = [F.conv2d(x[i].unsqueeze(0), w[i], bias=None,
                          stride=1, padding='same') for i in range(N)]
            y = torch.cat(y, dim=0)
        else:
            y = F.conv2d(x, w[0], bias=None, stride=1, padding='same')

        y = self.subsequent_layer(y)

        return y


class plExtractor(nn.Module):
    def __init__(self, device):
        super(plExtractor, self).__init__()
        
        self.device = device
        
        numc = 16
        
        self.re1c = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=1,out_chan=numc,)

        self.re1 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=1,out_chan=numc,)
        self.re2 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*4,out_chan=numc*2,)
        self.re3 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*2*4,out_chan=numc*4,)
        self.re4 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*4*4,out_chan=numc*8,)
        
        self.rb = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*8*4,out_chan=numc*16,)

        self.rd1 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*16)//4 + numc*8,out_chan=numc*8,)
        self.rd2 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*8)//4 + numc*4,out_chan=numc*4,)
        self.rd3 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*4)//4 + numc*2,out_chan=numc*2,)
        self.rd4 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*2)//4 + numc,out_chan=numc,)

        self.out = MoDEConv(5, 1, numc, 2, kernel_size=5, padding='same')

        self.dw = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, batch_first=True)
        # self.autobot = nn.TransformerEncoder(encoder_layer, 1)

        # self.out = conv(numc, 2, 1, 1, 0)
        self.ln = nn.LayerNorm(normalized_shape=512)
        self.s = nn.Sigmoid()

        self.dotmat = nn.Parameter(torch.empty(2,128, 1))
        nn.init.kaiming_normal_(
            self.dotmat, mode='fan_out', nonlinearity='relu')

    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, 1))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(self.device)

    def forward(self, x):

        
        cmap = 0#x[:,1]
        x = x[:,0]
        
        t = self.one_hot_task_embedding(torch.zeros(x.shape[0], dtype=int))
        #_, cmap = self.re1c(cmap,t)
        x1j, x1 = self.re1(x, t)  # [b,64*4,256,64]
        
        x2 = self.dw(x1+cmap)
        _, x2 = self.re2(x2, t)  # [b,64*4,256,64]
        x3 = self.dw(x2)
        _, x3 = self.re3(x3, t)  # [b,64*4,256,64]
        x4 = self.dw(x3)
        _, x4 = self.re4(x4, t)  # [b,64*4,256,64]
        lat = self.dw(x4)
        
        _, lat = self.rb(lat,t)
        # lat = self.autobot(lat.reshape([-1,256,256])).reshape([-1,256,32,8])
        
        lat = self.up(lat)
        _, x4 = self.rd1(torch.cat([lat, x4], dim=1),t)  # [b,128,64,16]
        x4 = self.up(x4)
        _, x3 = self.rd2(torch.cat([x4, x3], dim=1),t)  # [b,64,128,32]
        x3 = self.up(x3)
        _, x2 = self.rd3(torch.cat([x3, x2], dim=1),t)  # [b,32,256,64]
        x2 = self.up(x2)

        _, x1 = self.rd4(torch.cat([x2, x1+cmap], dim=1), t)

        x = self.out(x1, t)

        temp=[]
        for chan in range(0,2):
            dotmat = self.dotmat[chan].repeat(x.size(0),1,1)
            temp.append( torch.bmm(x[:,chan], dotmat).squeeze(2).unsqueeze(1))

        x = torch.stack(temp,dim=1)#.clamp(0,1)
        # x = self.autobot(self.ln(x.squeeze(2))).unsqueeze(2).clamp(0,1)

        x = self.s(x)
        return x, x
