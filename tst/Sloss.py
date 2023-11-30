import torch
import torch.nn as nn
from torch.nn import functional as F


class SLoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        \Delta_T = \sqrt{\int (y_{est}^T - y^T)^2}

        \Delta_Q = \sqrt{\int (y_{est}^Q - y^Q)^2}

        loss = log(1 + \Delta_T) + \\alpha \cdot log(1 + \Delta_Q)

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        #
        # self.alpha = alpha
        self.reduction = reduction
        self.base_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                s_true: torch.Tensor,
                r_pred: torch.Tensor,
                w_pred: torch.Tensor) -> torch.Tensor:
        # bn = w_pred.shape[0]
        wn = w_pred.shape[1]
        rn = r_pred.shape[1]
        sn = wn+rn-1
        wn2 = int(wn/2.0)
        # wl = F.pad(w_pred,[0,0,0,sn-wn],'constant',0)
        # rl = F.pad(r_pred, [0, 0, 0, sn - rn], 'constant', 0)
        Fw = torch.fft.fft(w_pred, sn, dim=1)
        Fr = torch.fft.fft(r_pred, sn, dim=1)
        Fs = Fw*Fr
        s_pred = torch.real(torch.fft.ifft(Fs,dim=1))
        s_pred = s_pred[:, wn2:-wn2, :]
        return self.base_loss(s_pred, s_true)


        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        # delta_Q = self.base_loss(y_pred[..., :-1], y_true[..., :-1])
        # delta_T = self.base_loss(y_pred[..., -1], y_true[..., -1])
        #
        # if self.reduction == 'none':
        #     delta_Q = delta_Q.mean(dim=(1, 2))
        #     delta_T = delta_T.mean(dim=(1))

        # return torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)
