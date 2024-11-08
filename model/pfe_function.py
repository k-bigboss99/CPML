
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import utils
from .TFA import TFA

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)



def pfe_heart_x(self,x,output_size):

    [batch, channel, length, width, height] = x.shape
    imnet = self.mlp_x
    coord = utils.make_coord([64,64]).cuda(device=self.device_ids[0])

    cell = torch.ones_like(coord)
    coord = coord.expand(batch*length,-1,-1)
    cell[:, 0] *= 2 / output_size
    cell[:, 1] *= 2 / output_size
    cell = cell.expand(batch*length,-1,-1)
    x = x.permute(0,2,1,3,4).contiguous().view(-1,channel,width,height)
    ret = self.query_rgb(x, coord,cell=cell,imnet=imnet)
    ret = ret.permute(0,2,1).contiguous().view(batch*length,channel,output_size,output_size).view(batch,length,channel,output_size,output_size)

    return ret

def pfe_heart_y(self,x,output_size):
    [batch, channel, length, width, height] = x.shape
    imnet = self.mlp_y
    coord = utils.make_coord([64,64]).cuda(device=self.device_ids[0])

    cell = torch.ones_like(coord)
    coord = coord.expand(batch*length,-1,-1)
    cell[:, 0] *= 2 / output_size
    cell[:, 1] *= 2 / output_size
    cell = cell.expand(batch*length,-1,-1)
    x = x.permute(0,2,1,3,4).contiguous().view(-1,channel,width,height)
    ret = self.query_rgb(x, coord,cell=cell,imnet=imnet)
    ret = ret.permute(0,2,1).contiguous().view(batch*length,channel,output_size,output_size).view(batch,length,channel,output_size,output_size)
    return ret

def query_rgb(self, x, coord, cell=None, imnet=None):
    feat = x
    imnet = imnet
    feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])# feature unfolding
    coord_ = coord.clone()


    q_feat = F.grid_sample(
        feat, coord_.flip(-1).unsqueeze(1),
        mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)


    rel_cell = cell.clone()
    rel_cell[:, :, 0] *= feat.shape[-2]
    rel_cell[:, :, 1] *= feat.shape[-1]
    inp = torch.cat([q_feat, rel_cell], dim=-1)

    bs, q = coord.shape[:2]
    pred = imnet(inp.view(bs * q, -1)).view(bs, q, -1)

    return pred


