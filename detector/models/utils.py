from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    # ind son los índices de las 1000 detecciones con mejor score
    # torch.Size([1, 2, 256, 144]) reg y torch.Size([1, 8, 256, 144]) vertices
    feat = feat.permute(0, 2, 3, 1).contiguous() #torch tensor que reacomodan y pone de manera contugua en memoria
    #  torch.Size([1, 256, 144, 2]) para reg y torch.Size([1, 256, 144, 8]) para polygon_vertices
    #https://pytorch.org/docs/stable/generated/torch.permute.html
    #https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
    feat = feat.view(feat.size(0), -1, feat.size(3)) # Returns a new tensor with the same data as the self tensor
    # but of a different shape.
    # torch.Size([1, 36864, 2]) para reg y torch.Size([1, 36864, 8])
    feat = _gather_feat(feat, ind)
    #torch.Size([1, 1000, 2]) para reg y torch.Size([1, 1000, 8]) para

    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)