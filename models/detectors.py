import torch
import torch.nn as nn
import torch.nn.functional as F
from models.torch_utils import join
from functorch import vmap
from toolz.curried import partial

class TramerTransform(nn.Module):
    """
    Given a trained classifier, uses the Tramer transformation to create a detector.
    The adversarial attack (specified by the attacker) is a proxy for the robustness
    criterion used in Tramer's paper (https://arxiv.org/abs/2107.11630).
    """
    def __init__(self, model, attacker, invert=False, **kwargs):
        super().__init__()
        self.model = model
        self.attacker = attacker

        self.invert = invert

    def forward(self, x):
        predictions_x = self.model(x).argmax(dim=1)

        _, x_adv, _ = self.attacker(
            self.model,
            (x, predictions_x),
        )
        predictions_x_adv = self.model(x_adv).argmax(dim=1)

        # Negative prediction signaling rejection
        if self.invert:
            predictions = predictions_x
            predictions[predictions_x == predictions_x_adv] = -1
        else:
            predictions = predictions_x
            predictions[predictions_x != predictions_x_adv] = -1

        return predictions.long()


