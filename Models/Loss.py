import torch
from torch import nn
from torch.nn import MSELoss, L1Loss, HuberLoss
import random
from utils import decimal_to_pentanary


class LossFunction(nn.Module):
    def __init__(
        self,
        MSE_importance: int = 1,
        MAE_importance: int = 1,
        Huber_importance: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super(LossFunction, self).__init__()
        self.MSE_Mask = (
            torch.tensor([1, 0, 0, 1, 1, 1, 0, 0], device=device) * MSE_importance
        )
        self.MAE_Mask = (
            torch.tensor([0, 1, 0, 0, 0, 0, 1, 1], device=device) * MAE_importance
        )
        self.Huber_Mask = (
            torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], device=device) * Huber_importance
        )
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


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, specialCase: str):
        new_targets = []
        for index, case_type in enumerate(specialCase):
            current_classification = decimal_to_pentanary(
                torch.argmax(outputs[index], dim=0).item()
            )
            calm_stormy_val = int(current_classification[0])
            clear_cloudy_val = int(current_classification[1])
            dry_wet_val = int(current_classification[2])
            cold_hot_val = int(current_classification[3])
            if case_type == "rain":
                dry_wet_val = random.randint(1, 4)
                calm_stormy_val = (
                    3
                    if dry_wet_val == 2
                    else (4 if dry_wet_val >= 3 else calm_stormy_val)
                )
            elif case_type == "rime":
                cold_hot_val = 0
            elif case_type == "snow":
                cold_hot_val = 0
                dry_wet_val = random.randint(1, 2)
            elif case_type == "lightning":
                calm_stormy_val = random.randint(2, 4)
            else:
                new_targets.append(targets[index])
                continue

            new_targets.append(
                cold_hot_val
                + 5 * dry_wet_val
                + 25 * clear_cloudy_val
                + 125 * calm_stormy_val
            )
        new_targets = torch.tensor(new_targets).to(outputs.device)

        return self.criterion(outputs, new_targets)
