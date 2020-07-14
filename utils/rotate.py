import torch
import torch.nn as nn
import torch.optim as optim

EPOCH = 1000


def get_constrain(weight: torch.Tensor):
    det_loss = weight.det() ** 2
    lc_loss = ((weight.norm(dim=0) - 1) ** 2).sum() + ((weight.norm(dim=1) - 1) ** 2).sum()
    return det_loss + lc_loss


def rotate_to(pos: torch.Tensor, fit_pos: torch.Tensor) -> torch.Tensor:
    linear = nn.Linear(3, 3, bias=True)
    optimizer = optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-2)

    for i in range(EPOCH):
        optimizer.zero_grad()
        new_pos = linear(pos)
        fit_loss = (new_pos - fit_pos).norm()
        c_loss = get_constrain(linear.weight)
        if i == EPOCH - 1:
            print('fit loss:', fit_loss.item())
            print('constraint loss:', c_loss.item())
        loss = fit_loss + c_loss * 10
        loss.backward()
        optimizer.step()

    # for name, param in linear.named_parameters():
    #     print(name, ":", param)
    return linear(pos).detach()
