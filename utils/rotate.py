import torch
import torch.nn as nn
import torch.optim as optim

EPOCH = 1000


def get_constrain(weight: torch.Tensor):
    det_loss = (weight.det() - 1) ** 2
    lc_loss = ((weight.norm(dim=0) - 1) ** 2).sum() + ((weight.norm(dim=1) - 1) ** 2).sum()
    return det_loss + lc_loss


def rotate_to(pos: torch.Tensor, fit_pos: torch.Tensor) -> torch.Tensor:
    linear = nn.Linear(3, 3, bias=True)
    optimizer = optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-2)

    loss_1 = 1e8
    for i in range(EPOCH):
        optimizer.zero_grad()
        new_pos = linear(pos)
        fit_loss = (new_pos - fit_pos).norm()
        c_loss = get_constrain(linear.weight)
        if i == EPOCH - 1:
            print('fit loss:', fit_loss.item())
            print('constraint loss:', c_loss.item())
        loss = fit_loss + c_loss * 1
        loss.backward()
        optimizer.step()
        loss_1 = loss.item()

    new_pos_1 = linear(pos).detach()

    pos = pos * torch.tensor([-1, 1, 1], dtype=torch.float32)
    linear = nn.Linear(3, 3, bias=True)
    optimizer = optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-2)

    loss_2 = 1e8
    for i in range(EPOCH):
        optimizer.zero_grad()
        new_pos = linear(pos)
        fit_loss = (new_pos - fit_pos).norm()
        c_loss = get_constrain(linear.weight)
        if i == EPOCH - 1:
            print('fit loss:', fit_loss.item())
            print('constraint loss:', c_loss.item())
        loss = fit_loss + c_loss * 1
        loss.backward()
        optimizer.step()
        loss_2 = loss.item()

    new_pos_2 = linear(pos).detach()

    # for name, param in linear.named_parameters():
    #     print(name, ":", param)
    return new_pos_1 if loss_1 < loss_2 else new_pos_2
