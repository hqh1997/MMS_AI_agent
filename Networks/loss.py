import numpy as np
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class NegativeLogLikelihoodSurvivalLoss:
    def __call__(self, hazards, S, Y, c, alpha=0.15, eps=1e-7):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)
        c = c.view(batch_size, 1).float()
        S_padded = torch.cat([torch.ones_like(c), S], 1)
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
            torch.gather(hazards, 1, Y).clamp(min=eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss
        
class CoxSurvivalLoss:
    def __call__(self, hazards, S, c):
        # https://github.com/traversc/cox-nnet
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        return loss_cox
