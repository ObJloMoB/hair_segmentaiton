import torch

def iou(pred, mask):
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    Union = torch.where(pred > mask, pred, mask)
    Overlep = torch.mul(pred, mask)
    o = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
    return o