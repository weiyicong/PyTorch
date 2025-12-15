import torch

x_data=torch.Tensor([1.0],[2.0],[3.0])
y_data=torch.Tensor([2.0],[4.0],[6.0])

class LinerModel(torch.nn.module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(1,1)