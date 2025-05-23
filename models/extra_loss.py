import torch
import torch.nn.functional as F

def kl_loss(dist,labels):
    """
    This is a experimental implementation of KL-divergence for distance distributions.
    """
    gt_fuse_pts0 = labels['gt_fuse_pts0'][0][:labels['gt_fuse_num']]
    gt_fuse_pts1 = labels['gt_fuse_pts1'][0][:labels['gt_fuse_num']]
    for i in range(labels['gt_fuse_num']):
        if i == 0:
            gt_dist = torch.cdist(gt_fuse_pts0[i].unsqueeze(0), gt_fuse_pts1[i].unsqueeze(0))
        else:
            gt_dist = torch.cat((gt_dist, torch.cdist(gt_fuse_pts0[i].unsqueeze(0), gt_fuse_pts1[i].unsqueeze(0))), dim=0)
    num_bins = 10
    min_val = min(dist.min(), gt_dist.min())
    max_val = max(dist.max(), gt_dist.max())
    pred_hist = torch.histc(dist, bins=num_bins, min=min_val, max=max_val)
    pred_dist = pred_hist / pred_hist.sum()
    gt_hist = torch.histc(gt_dist, bins=num_bins, min=min_val, max=max_val)
    gt_dist_prob = gt_hist / gt_hist.sum()
    epsilon = 1e-8
    kl_div = F.kl_div((pred_dist + epsilon).log(), gt_dist_prob.cuda() + epsilon, reduction='sum')

    kl_div = kl_div# + norm*0.1

    return kl_div.item()
