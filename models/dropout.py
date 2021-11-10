import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import Tensor

def dropout(_input, p, training, inplace, mask=None):
    if mask is None:
        if training:
            mask = torch.bernoulli(_input.data.new(_input.data.size()).fill_(p)).div(1-p)
        else:
            mask = torch.ones_like(_input)

    if inplace:
        res = _input.mul_(mask)
    else:
        res = _input.mul(mask)

    return res, mask

class _DropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.fix_mask = False

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout(_DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.
    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .
    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    Examples::
        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input: Tensor) -> Tensor:
        res, mask = dropout(input, self.p, self.training, self.inplace, mask=self.curr_mask if self.fix_mask else None)
        self.curr_mask = mask.detach()
        return res
