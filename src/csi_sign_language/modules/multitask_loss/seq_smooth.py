import torch
from torch import Tensor


def t_mse(y: Tensor, tau, t_length=None):
    """
    from "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation"
    duing with spike phonemno
    @param y: [..., T, C] should be logged softmax
    @param tau: float
    @param t_length: [...] valid t_length
    @param reduce: str the reduction method
    @return: [...]
    """
    C, T = y.shape[-1], y.shape[-2]
    assert T > 1, "temporal dimension should be greater than 1"

    y_t = y
    y_tp = torch.cat((y[..., 0:1, :], y[..., :-1, :]), dim=-2).detach()

    delta_t = torch.abs(y_t - y_tp)
    delta_t_hat = torch.where(delta_t < tau, tau, delta_t)

    loss = torch.sum(delta_t_hat**2, dim=(-1)) / C

    if t_length is not None:
        mask = torch.arange(T, device=y.device).expand(
            y.shape[:-2] + (T,)
        ) < t_length.unsqueeze(-1)
        mask = torch.zeros_like(mask, device=y.device).masked_fill(mask, 1).float()
        loss = loss * mask

    loss = loss[..., 1:]  # remove the T=0, for it is meaningless

    return loss.sum() / T


if __name__ == "__main__":
    y = torch.randn(2, 10, 5, requires_grad=True) * 10
    length = torch.tensor([8, 7])
    y = torch.nn.functional.log_softmax(y, dim=-1)
    tau = 0.1
    result = t_mse(y, tau, t_length=length)
    print(result)
    y.retain_grad()
    result.backward()
    print(y.grad)
    print(result.shape)
