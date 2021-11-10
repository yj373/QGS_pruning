import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

module = model.conv1

prune_method = prune.L1Unstructured(amount=0.5)
old_weight = module.weight.data
pruned = prune_method.prune(module.weight.data)
print(old_weight)
print(pruned)
print(torch.norm((pruned - module.weight.data).view(-1), 1))
print(torch.norm((pruned - old_weight).view(-1), 1))
# mask = prune_method.compute_mask(module.weight, default_mask=torch.ones_like(module.weight))
# print('>>>>>>>>>>>>>>>original>>>>>>>>>>>>>>>>>')
# print(module.weight)
# print('>>>>>>>>>>>>l1>>>>>>>>>>>')
# print(mask)
# # prune.custom_from_mask(module, name='weight', mask=mask)
# prune_method.apply(module, name='weight', amount=prune_method.amount)
# print(module.weight)
# old_weight = module.state_dict()['weight_orig'].data
# prune.remove(module, name='weight')
# print('>>>>>>>custom>>>>>>>>>>>>>>>>>>')
# module.weight.data = old_weight
# # prune.remove(module, name='weight')
# k = torch.ones_like(mask) * 0.7
# new_mask = torch.where(mask == 0, k, mask)
# print(new_mask)
# prune.custom_from_mask(module, name='weight', mask=new_mask)
# print(module.weight)
    
