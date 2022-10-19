import math
import torch


__all__ = [
    "check_task_1",
    "check_task_2",
    "check_task_3"
]


def check_task_1(pred_res, pred_attn):
    true_res  = torch.tensor([[ 0.1212,  0.5156, -0.2394, -0.1912],
                             [ 0.0999,  0.5376, -0.2558, -0.1143],
                             [ 0.1348,  0.5492, -0.3327, -0.3267]])

    true_attn = torch.tensor([[0.3017, 0.3098, 0.3884],
                              [0.2451, 0.3801, 0.3748],
                              [0.2938, 0.2293, 0.4769]])
    
    assert torch.allclose(pred_res, true_res, rtol=1e-4, atol=1e-4), "\033[91m Something is wrong :("
    assert torch.allclose(pred_attn, true_attn, rtol=1e-4, atol=1e-4), "\033[91m Something is wrong :("
    print('\033[92m Well done :)')


def check_task_2(pred_attn_output):
    true_attn_output = torch.tensor([[[-0.0336,  0.1684, -0.1582,  0.3691],
                                      [-0.1676,  0.2687, -0.2593,  0.3329],
                                      [ 0.0423,  0.1611, -0.1277,  0.4158]]])

    assert torch.allclose(pred_attn_output, true_attn_output, rtol=1e-4, atol=1e-4), "\033[91m Something is wrong :("
    print('\033[92m Well done :)')


def check_task_3(pred_pe):
    true_pe = torch.zeros(128, 64)
    position = torch.arange(0, 128, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, 64, 2).float() * (-math.log(10000.0) / 64))
    true_pe[:, 0::2] = torch.sin(position * div_term)
    true_pe[:, 1::2] = torch.cos(position * div_term)
    assert torch.allclose(pred_pe, true_pe, rtol=1e-4, atol=1e-4), "\033[91m Something is wrong :("
    print('\033[92m Well done :)')
