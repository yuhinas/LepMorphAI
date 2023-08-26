import numpy as np
import torch

def codes_to_no_neg (codes, is_cuda=False):
    if is_cuda:
        codes_pos = codes.clone().detach()
        codes_neg = 0 - codes.clone().detach()
        codes_no_neg = torch.cat([codes_pos, codes_neg], dim=1)
        codes_no_neg[codes_no_neg < 0] = 0
    else:
        codes_pos = codes.copy()
        codes_neg = 0 - codes.copy()
        codes_no_neg = np.concatenate([codes_pos, codes_neg], axis=1)
        codes_no_neg[codes_no_neg < 0] = 0

    return codes_no_neg


def codes_to_with_neg (codes, is_cuda=False):
    if is_cuda:
        code_dim = int(codes.size(1) / 2)
    else:
        code_dim = int(codes.shape[1] / 2)

    codes_pos = codes[:,:code_dim]
    codes_neg = codes[:, code_dim:]

    return codes_pos - codes_neg

