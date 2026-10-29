# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable


def average_precision(recalls, precisions, mode='area'):
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_det_cls(pred, gt, iou_thr=None):
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[-1] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    pred_boxes = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 9), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box.tensor
            pred_boxes.append(box)
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]
    pred_boxes = [pred_boxes[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        # Select the ground-truth box with the highest IoU.
        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if R['det'][iou_idx][jmax] == -1:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = d
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        eps = 1e-5
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / (npos + eps)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        accuracy = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, accuracy)

        # compute ate
        tp_num = 0
        dist_sum = 0.
        iou_sum = 0.
        angle_sum = 0.
        sum = 0
        for img_id in class_recs.keys():
            R = class_recs[img_id]
            for gt_box_id, pred_bbox_id in enumerate(R['det'][iou_idx]):
                sum += 1
                if pred_bbox_id != -1:
                    tp_num += 1
                    pred_tensor = torch.squeeze(pred_boxes[pred_bbox_id].tensor)
                    gt_tensor = torch.squeeze(R['bbox'][gt_box_id].tensor)
                    dist_sum += ((pred_tensor[0] - gt_tensor[0]) ** 2 + (pred_tensor[1] - gt_tensor[1]) ** 2) ** 0.5
                    iou_sum += ious[pred_bbox_id][gt_box_id]
                    angle_sum += abs(pred_tensor[6] - gt_tensor[6])
        print(iou_idx, tp_num, sum)
        ate = np.array([dist_sum / (tp_num + eps)])
        ase = np.array([1 - iou_sum / (tp_num + eps)])
        aoe = np.array([angle_sum / (tp_num + eps)])
        nds = np.array([ap[-1] * 0.5 + (1 - ate / 2 + 1 - ase + 1 - aoe / np.pi * 6 + 1 - accuracy[-1]) / 8])

        ret.append((recall, ap, ate, ase, aoe, accuracy, nds))

    return ret


def eval_map_recall(pred, gt, ovthresh=None):
    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    acc = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]
    ate = [{} for i in ovthresh]
    ase = [{} for i in ovthresh]
    aoe = [{} for i in ovthresh]
    nds = [{} for i in ovthresh]

    for classname in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if classname in pred:
                (recall[iou_idx][classname],
                 ap[iou_idx][classname],
                 ate[iou_idx][classname],
                 ase[iou_idx][classname],
                 aoe[iou_idx][classname],
                 acc[iou_idx][classname],
                 nds[iou_idx][classname]) = ret_values[classname][iou_idx]
            else:
                recall[iou_idx][classname] = np.zeros(1)
                acc[iou_idx][classname] = np.zeros(1)
                ap[iou_idx][classname] = np.zeros(1)
                ate[iou_idx][classname] = np.zeros(1)
                ase[iou_idx][classname] = np.zeros(1)
                aoe[iou_idx][classname] = np.zeros(1)
                nds[iou_idx][classname] = np.zeros(1)

    return recall, ap, ate, ase, aoe, acc, nds

def class2index(gt_names, label2cat):
    class_ids = []
    for name in gt_names:
        if name in label2cat:
            class_ids.append(label2cat[name])
        else:
            class_ids.append(9)
    return np.array(class_ids)


def indoor_eval(gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None,
                box_type_3d=None,
                box_mode_3d=None,
                point_cloud_range=None):
    if point_cloud_range is None:
        point_cloud_range = np.array([0, -51.2, -5.0, 51.2, 51.2, 3.0])
    assert len(dt_annos) == len(gt_annos)
    className2index = {label2cat[i]:i for i in label2cat.keys()}
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    for img_id in range(len(dt_annos)):
        # parse detected annotations
        det_anno = dt_annos[img_id]['pts_bbox']
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

        # parse gt annotations
        gt_anno = gt_annos[img_id]
        gt_anno['gt_num'] = len(gt_anno['gt_names'])
        if gt_anno['gt_num'] != 0:
            gt_boxes = box_type_3d(
                gt_anno['gt_boxes'],
                box_dim=gt_anno['gt_boxes'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            labels_3d = class2index(gt_anno['gt_names'], className2index)
            mask = gt_boxes.in_range_bev(point_cloud_range[[0, 1, 3, 4]])
            gt_boxes = gt_boxes[mask]
            labels_3d = labels_3d[mask.numpy().astype(bool)]
            gt_boxes.limit_yaw(offset=0.5, period=2 * np.pi)
        else:
            gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            labels_3d = np.array([], dtype=np.int64)

        for i in range(len(labels_3d)):
            label = labels_3d[i]
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    rec, ap, ate, ase, aoe, aae, nds = eval_map_recall(pred, gt, metric)
    ret_dict = dict()
    header = ['classes']
    table_columns = [[label2cat[label] for label in ap[0].keys()] + ['Overall']]

    for i, iou_thresh in enumerate(metric):
        ## ap
        header.append(f'AP_{iou_thresh:.2f}')
        ap_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(ap[i][label][0])
            ap_list.append(ap[i][label][-1])
        ret_dict[f'mAP_{iou_thresh:.2f}'] = float(np.mean(ap_list))

        table_columns.append(list(map(float, ap_list)))  # Append each class result.
        table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]  # Append the aggregate mAP.
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]  # Format each value to four decimal places.

        ## ar
        header.append(f'AR_{iou_thresh:.2f}')
        ar_list = []
        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_AR_{iou_thresh:.2f}'] = float(rec[i][label][-1])
            ar_list.append(rec[i][label][-1])
        ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.mean(ar_list))

        table_columns.append(list(map(float, ar_list)))
        table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        ## ate
        header.append(f'ATE_{iou_thresh:.2f}')
        ate_list = []
        for label in ate[i].keys():
            ret_dict[f'{label2cat[label]}_ATE_{iou_thresh:.2f}'] = float(ate[i][label][-1])
            ate_list.append(ate[i][label][-1])
        ret_dict[f'mATE_{iou_thresh:.2f}'] = float(np.mean(ate_list))

        table_columns.append(list(map(float, ate_list)))
        table_columns[-1] += [ret_dict[f'mATE_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        ## ase
        header.append(f'ASE_{iou_thresh:.2f}')
        ase_list = []
        for label in ase[i].keys():
            ret_dict[f'{label2cat[label]}_ASE_{iou_thresh:.2f}'] = float(ase[i][label][-1])
            ase_list.append(ase[i][label][-1])
        ret_dict[f'mASE_{iou_thresh:.2f}'] = float(np.mean(ase_list))

        table_columns.append(list(map(float, ase_list)))
        table_columns[-1] += [ret_dict[f'mASE_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        ## aoe
        header.append(f'AOE_{iou_thresh:.2f}')
        aoe_list = []
        for label in aoe[i].keys():
            ret_dict[f'{label2cat[label]}_AOE_{iou_thresh:.2f}'] = float(aoe[i][label][-1])
            aoe_list.append(aoe[i][label][-1])
        ret_dict[f'mAOE_{iou_thresh:.2f}'] = float(np.mean(aoe_list))

        table_columns.append(list(map(float, aoe_list)))
        table_columns[-1] += [ret_dict[f'mAOE_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        ## aae
        header.append(f'AAE_{iou_thresh:.2f}')
        aae_list = []
        for label in aae[i].keys():
            ret_dict[f'{label2cat[label]}_AAE_{iou_thresh:.2f}'] = float(aae[i][label][-1])
            aae_list.append(aae[i][label][-1])
        ret_dict[f'mAAE_{iou_thresh:.2f}'] = float(np.mean(aae_list))

        table_columns.append(list(map(float, aae_list)))
        table_columns[-1] += [ret_dict[f'mAAE_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        ## nds
        header.append(f'NDS_{iou_thresh:.2f}')
        nds_list = []
        for label in nds[i].keys():
            ret_dict[f'{label2cat[label]}_NDS_{iou_thresh:.2f}'] = float(nds[i][label][-1])
            nds_list.append(nds[i][label][-1])
        ret_dict[f'mNDS_{iou_thresh:.2f}'] = float(np.mean(nds_list))

        table_columns.append(list(map(float, nds_list)))
        table_columns[-1] += [ret_dict[f'mNDS_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict
