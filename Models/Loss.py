import torch
from torch import nn
from torch.nn import MSELoss, L1Loss, HuberLoss

class LossFunction(nn.Module):
    def __init__(self, MSE_importance:int = 1, MAE_importance:int =1, Huber_importance:int = 1 ,
                 device:torch.device = torch.device("cpu")):
        super(LossFunction, self).__init__()
        self.MSE_Mask = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0],device=device) * MSE_importance
        self.MAE_Mask = torch.tensor([0, 1, 0, 0, 0, 0, 1, 1],device=device) * MAE_importance
        self.Huber_Mask = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], device=device) * Huber_importance
        self._MSEFunction = MSELoss()
        self._MAEFunction = L1Loss()
        self._HuberFunction = HuberLoss()

    def _MSE(self, output, target):
        return self._MSEFunction(output, target) * self.MSE_Mask

    def _MAE(self, output, target):
        return self._MAEFunction(output, target) * self.MAE_Mask

    def _Huber(self, output, target):
        return self._HuberFunction(output, target) * self.Huber_Mask

    def forward(self, output, target):
        MSE_loss = self._MSE(output, target)
        MAE_loss = self._MAE(output, target)
        Huber_loss = self._Huber(output, target)
        combined_loss = MSE_loss + MAE_loss + Huber_loss
        return torch.mean(combined_loss)