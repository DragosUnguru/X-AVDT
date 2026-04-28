import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # targets: (B,) in {0,1}, logits: (B,)
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            logits, targets = logits[mask], targets[mask]

        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')  # (B,)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt)**self.gamma * bce
        return loss.mean()


# https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_name='ContrastiveLoss', embedding_size=1024, pos_margin=0.0, neg_margin=1.0,
                 memory_size=None, use_miner=False, num_classes=2, tau=0.5):
        super(CombinedLoss, self).__init__()
        self.loss_name = loss_name
        self.embedding_size = embedding_size
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.memory_size = memory_size
        self.use_miner = use_miner
        self.num_classes = num_classes

        if loss_name == 'TripletMarginLoss':
            self.loss_fn = losses.TripletMarginLoss(smooth_loss=True)
        elif loss_name == 'ArcFaceLoss':
            self.loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
        elif loss_name == 'SubCenterArcFaceLoss':
            self.loss_fn = losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
        elif loss_name == 'CircleLoss':
            self.loss_fn = losses.CircleLoss()
        elif loss_name == 'NTXentLoss':
            self.loss_fn = losses.NTXentLoss(temperature=tau)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        else:
            self.loss_fn = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)

        miner = miners.MultiSimilarityMiner() if use_miner else None
        if memory_size is not None:
            self.loss_fn = losses.CrossBatchMemory(self.loss_fn, embedding_size=embedding_size, memory_size=memory_size, miner=miner)

    def forward(self, embeddings, labels):
        loss = self.loss_fn(embeddings, labels)

        return loss
