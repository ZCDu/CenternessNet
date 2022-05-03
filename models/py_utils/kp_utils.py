import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .utils import convolution, residual

INF = 1e8


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


def make_merge_layer(dim):
    return MergeUp()


def make_tl_layer(dim):
    return None


def make_br_layer(dim):
    return None


def make_ct_layer(dim):
    return None


def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                         nn.Conv2d(curr_dim, out_dim, (1, 1)))


def make_inter_layer(dim):
    return residual(3, dim, dim)


def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel),
                                    stride=1,
                                    padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _decode(tl_heat,
            br_heat,
            tl_tag,
            br_tag,
            tl_regr,
            br_regr,
            ct_heat,
            ct_regr,
            centerness_map,
            K=100,
            kernel=1,
            ae_threshold=1,
            num_dets=1000):
    batch, cat, height, width = tl_heat.size()
    #print("py_utils kp_utils decode heat:", tl_heat.shape)
    #print("py_utils kp_utils decode centerness_map:", centerness_map.shape)
    # add
    centerness_map = centerness_map.sigmoid()
    #ct_heat = ct_heat * centerness_map
    #print('ct_heat:', ct_heat.shape)

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    ct_heat = torch.sigmoid(ct_heat)
    ct_heat = ct_heat * centerness_map

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    ct_heat = _nms(ct_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
    ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)
        ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)
        ct_regr = ct_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        ct_xs = ct_xs + ct_regr[..., 0]
        ct_ys = ct_ys + ct_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    #width = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    #height = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    ct_xs = ct_xs[:, 0, :]
    ct_ys = ct_ys[:, 0, :]

    center = torch.cat([
        ct_xs.unsqueeze(2),
        ct_ys.unsqueeze(2),
        ct_clses.float().unsqueeze(2),
        ct_scores.unsqueeze(2)
    ],
                       dim=2)
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses],
                           dim=2)
    return detections, center


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred,
                                                       2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


def _ae_loss(tag0, tag1, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _regr_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def get_points(featmap_size, dtype, device, flatten=False):
    h, w = featmap_size
    stride = int(512 / 128)
    x_range = torch.arange(w, dtype=dtype, device=device)
    y_range = torch.arange(h, dtype=dtype, device=device)
    y, x = torch.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()
    points = torch.stack(
        (x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1) + stride // 2
    return points


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def get_targets(gt_bboxes,
                gt_labels,
                points=None,
                num_classes=80,
                regress_range=(-1, INF),
                center_sampling=False,
                center_sample_radius=1.5):
    #print('py_utils kp_utils points:', points)
    #print('py_utils kp_utils gt_bboxes:', gt_bboxes[0].shape)
    labels_list, bbox_targets_list = multi_apply(
        get_targets_single,
        gt_bboxes,
        gt_labels,
        points=points,
        regress_range=regress_range,
        num_classes=num_classes,
        center_sampling=center_sampling,
        center_sample_radius=center_sample_radius)
    return labels_list, bbox_targets_list


def get_targets_single(gt_bboxes, gt_labels, points, regress_range,
                       num_classes, center_sampling, center_sample_radius):
    # 需要一个regress_range来限制越界
    expanded_regress_range = points.new_tensor(regress_range)[None].expand_as(
        points)
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        gt_labels = gt_labels.new_full((num_points, ), num_classes)
        gt_bboxes = gt_bboxes.new_zeros((num_points, 4))
        return gt_labels.cuda(), gt_bboxes.cuda()
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] -
                                                   gt_bboxes[:, 1])
    areas = areas[None].repeat(num_points, 1)
    regress_range = expanded_regress_range[:, None, :].expand(
        num_points, num_gts, 2)
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4).cuda()
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_gts)
    ys = ys[:, None].expand(num_points, num_gts)

    # xs:torch.cuda.float32;gt_bboxes:torch.float32
    left = xs - gt_bboxes[..., 0]
    right = gt_bboxes[..., 2] - xs
    top = ys - gt_bboxes[..., 1]
    bottom = gt_bboxes[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)

    if center_sampling:
        radius = center_sample_radius
        center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        x_mins = center_xs - stride
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride
        center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins,
                                         gt_bboxes[..., 0])
        center_gts[..., 1] = torch.where(x_mins > gt_bboxes[..., 1], x_mins,
                                         gt_bboxes[..., 1])
        center_gts[..., 2] = torch.where(x_mins > gt_bboxes[..., 2], x_mins,
                                         gt_bboxes[..., 2])
        center_gts[..., 3] = torch.where(x_mins > gt_bboxes[..., 3], x_mins,
                                         gt_bboxes[..., 3])

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
    else:
        # condition1
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

    # condition2
    max_regress_distance = bbox_targets.max(-1)[0]
    inside_regress_range = (max_regress_distance >= regress_range[..., 0]) & (
        max_regress_distance <= regress_range[..., 1])

    areas[inside_gt_bbox_mask == 0] = INF
    areas[inside_regress_range == 0] = INF
    min_area, min_area_inds = areas.min(dim=1)

    labels = gt_labels[min_area_inds]
    labels[min_area == INF] = num_classes # set as BG
    bbox_targets = bbox_targets[range(num_points), min_area_inds]
    return labels.cuda(), bbox_targets


def centerness_target(pos_bbox_targets):
    left_right = pos_bbox_targets[:, [0, 2]]
    top_bottom = pos_bbox_targets[:, [1, 3]]
    centerness_target = (left_right.min(dim=-1)[0] / left_right.max(
        dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness_target)


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    # Caution: this function should only be used in RPN
    # in other files such as in ghm_loss, the _expand_binary_labels
    # is used for multi-class classification.
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred,
                                              label.float(),
                                              weight,
                                              reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice,
                                              target,
                                              reduction='mean')[None]


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss,
                              weight=weight,
                              reduction=reduction,
                              avg_factor=avg_factor)

    return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(cls_score,
                                                         label,
                                                         weight,
                                                         reduction=reduction,
                                                         avg_factor=avg_factor,
                                                         **kwargs)
        return loss_cls


def _centerness_loss(centerness,
                     gt_bboxes,
                     gt_labels,
                     img_meta=None,
                     gt_bboxes_ignore=None):
    num_classes = 80
    featmap_sizes = centerness.size()[-2:]
    num_imgs = centerness.size(0)
    loss_centerness = CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)
    # 在这里使得生成的点在gpu上, points=[16384,2]
    points = get_points(featmap_sizes, centerness.dtype, centerness.device)
    # labels=list(2)
    labels, bbox_targets = get_targets(gt_bboxes,
                                       gt_labels,
                                       points=points,
                                       num_classes=num_classes)
    flatten_centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
    # 需要知道batch的大小
    flatten_points = points.repeat(num_imgs, 1)
    #print('py_utils kp_utils labels:', labels)
    flatten_labels = torch.cat(labels)
    #print('py_utils kp_utils bbox_targets:', bbox_targets)
    flatten_bbox_targets = torch.cat(bbox_targets)
    # 背景类别设置
    bg_class_ind = num_classes
    pos_inds = ((flatten_labels >= 0) &
                (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
    num_pos = len(pos_inds)
    pos_centerness = flatten_centerness[pos_inds]
    if num_pos > 0:
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = centerness_target(pos_bbox_targets)
        centerness_loss = loss_centerness(pos_centerness,
                                          pos_centerness_targets)
    else:
        centerness_loss = pos_centerness.sum()
    return centerness_loss
