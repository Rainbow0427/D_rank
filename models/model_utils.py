import torch
import torch.nn as nn
import torch.nn.functional as F



def build_dynamic_basis_collection(groups, k_list, in_dim):
    """
    groups:  List[List[int]]   (e.g. [[0,1],[2,3],…])
    k_list:  List[int]         (e.g. [512,600,…], same length as groups)
    in_dim:  int               (hidden_size)
    returns: nn.ModuleDict mapping str(group[0]) -> nn.Linear(in_dim, k_i, bias=False)
    """
    assert len(groups) == len(k_list), "must supply one k per group"
    d = in_dim
    md = nn.ModuleDict()
    for grp, k in zip(groups, k_list):
        # we key by the first layer idx (just like ShareLlamaDecoderLayer expects)
        md[str(grp[0])] = nn.Linear(d, k, bias=False)
    return md

def build_basis_collection(groups, num_basis, nx):
    model_dict = torch.nn.ModuleDict()
    for group in groups:
        basis = Basis(num_basis, nx)
        for item in group:
            model_dict[str(item)] = basis
    return model_dict


class Basis(nn.Linear):
    def __init__(self, num_basis, nx):
        super().__init__(nx, num_basis, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())


class Coefficient(nn.Linear):
    def __init__(self, nf, num_basis, bias=False):
        super().__init__(num_basis, nf, bias=bias)
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = F.linear(x, self.weight, self.bias)
        x = x.view(size_out)
        return x

    def set_weight(self, weight, bias=None):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())
