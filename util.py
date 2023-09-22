import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
import os.path as osp
import os
import random
import numpy.linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cat(f_array):
    return np.concatenate(f_array, axis=0)

def L2_normalize_numpy(x, verbose=False):
    """Row normalization
    Args:
    x: a numpy matrix of shape N*D
    Returns:
    x: L2 normalized x
    """
    sqr_row_sum = LA.norm(x, axis=1, keepdims=True)
    iszero = sqr_row_sum <= 1e-7
    if verbose:
        print(f'There are {iszero.sum()} zero-padding feature(s).')
    sqr_row_sum[iszero] = 1  # XJ: avoid division by zero
    y = x / sqr_row_sum
    del x, sqr_row_sum, iszero
    return y


def cosine_similarity(f_x, f_y=None):
    f_xx_normalized = F.normalize(f_x, p=2, dim=-1)
    if f_y is None:
        f_yy_normalized = f_xx_normalized
    else:
        f_yy_normalized = F.normalize(f_y, p=2, dim=-1)
    f_yy_normalized_transpose = f_yy_normalized.transpose(1, 2)
    cosine_dis = torch.bmm(f_xx_normalized, f_yy_normalized_transpose)
    return cosine_dis

def compute_loss2(all_label_in_edge, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef=0.3, neg_weight=1, generation_weight=0.2):
    # n_pivot = len((modal_id[0]==2).nonzero())
    # n_pivot = len((modal_id[0]==0).nonzero())//n_pivot
    # n_pivot = 8
    # assert n_pivot==4
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==0]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    generation_weights = []
    for i in range(len(point_similarities)-1):
        generation_weights.append(generation_weight)
    generation_weights.append(1.0)
    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)*weight
        for (point_similarity, weight)
        in zip(point_similarities, generation_weights)]
    point_edge_loss_list = []
    point_loss = 0
    for i in range(len(total_edge_loss_generations_instance)):
        point_edge_loss = total_edge_loss_generations_instance[i]
        for j in range(3):
            selected_idxs = torch.nonzero(modal_id[0]==j)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            point_edge_loss_list.append(torch.mean(point_edge_loss[:,start:end, start:end]))
            point_loss += point_edge_loss_list[-1]

    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)*weight
        for (distribution_similarity, weight)
        in zip(distribution_similarities, generation_weights)]
    # for i in range(len(total_edge_loss_generations_distribution)):
    #     total_edge_loss_generations_distribution[i][ignore_mask<=0] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = 1
    total_edge_loss_generations = [
        distribution_loss_coeff * total_edge_loss_distribution
        for total_edge_loss_distribution in total_edge_loss_generations_distribution]
    
    # total_loss = 1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,:32])\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,32:64])+\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,64:])+\
    #     + dis_coef*point_loss/len(point_edge_loss_list)
    # total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0))*(all_label_in_edge.shape[1]/n_pivot) + dis_coef*point_loss/len(point_edge_loss_list)
    total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0)) + dis_coef*point_loss/len(point_edge_loss_list)

    return total_loss

def compute_loss2_face(all_label_in_edge, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef=0.3, neg_weight=1, generation_weight=0.2):
    # n_pivot = len((modal_id[0]==2).nonzero())
    # n_pivot = len((modal_id[0]==0).nonzero())//n_pivot
    # n_pivot = 8
    # assert n_pivot==4
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==0]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    generation_weights = []
    for i in range(len(point_similarities)-1):
        generation_weights.append(generation_weight)
    generation_weights.append(1.0)
    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)*weight
        for (point_similarity, weight)
        in zip(point_similarities, generation_weights)]
    point_edge_loss_list = []
    point_loss = 0
    for i in range(len(total_edge_loss_generations_instance)):
        point_edge_loss = total_edge_loss_generations_instance[i]
        for j in range(1):
            selected_idxs = torch.nonzero(modal_id[0]==j)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            point_edge_loss_list.append(torch.mean(point_edge_loss[:,start:end, start:end]))
            point_loss += point_edge_loss_list[-1]

    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)*weight
        for (distribution_similarity, weight)
        in zip(distribution_similarities, generation_weights)]
    # for i in range(len(total_edge_loss_generations_distribution)):
    #     total_edge_loss_generations_distribution[i][ignore_mask<=0] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = 1
    total_edge_loss_generations = [
        distribution_loss_coeff * total_edge_loss_distribution
        for total_edge_loss_distribution in total_edge_loss_generations_distribution]
    
    # total_loss = 1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,:32])\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,32:64])+\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,64:])+\
    #     + dis_coef*point_loss/len(point_edge_loss_list)
    # total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0))*(all_label_in_edge.shape[1]/n_pivot) + dis_coef*point_loss/len(point_edge_loss_list)
    total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0)) + dis_coef*point_loss/len(point_edge_loss_list)

    return total_loss

def compute_loss2_face_body(all_label_in_edge, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef=0.3, neg_weight=1, generation_weight=0.2):
    # n_pivot = len((modal_id[0]==2).nonzero())
    # n_pivot = len((modal_id[0]==0).nonzero())//n_pivot
    # n_pivot = 8
    # assert n_pivot==4
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==0]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    generation_weights = []
    for i in range(len(point_similarities)-1):
        generation_weights.append(generation_weight)
    generation_weights.append(1.0)
    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)*weight
        for (point_similarity, weight)
        in zip(point_similarities, generation_weights)]
    point_edge_loss_list = []
    point_loss = 0
    for i in range(len(total_edge_loss_generations_instance)):
        point_edge_loss = total_edge_loss_generations_instance[i]
        for j in range(2):
            selected_idxs = torch.nonzero(modal_id[0]==j)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            point_edge_loss_list.append(torch.mean(point_edge_loss[:,start:end, start:end]))
            point_loss += point_edge_loss_list[-1]

    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)*weight
        for (distribution_similarity, weight)
        in zip(distribution_similarities, generation_weights)]
    # for i in range(len(total_edge_loss_generations_distribution)):
    #     total_edge_loss_generations_distribution[i][ignore_mask<=0] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = 1
    total_edge_loss_generations = [
        distribution_loss_coeff * total_edge_loss_distribution
        for total_edge_loss_distribution in total_edge_loss_generations_distribution]
    
    # total_loss = 1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,:32])\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,32:64])+\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,64:])+\
    #     + dis_coef*point_loss/len(point_edge_loss_list)
    # total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0))*(all_label_in_edge.shape[1]/n_pivot) + dis_coef*point_loss/len(point_edge_loss_list)
    total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0)) + dis_coef*point_loss/len(point_edge_loss_list)

    return total_loss

def compute_loss_dis(all_label_in_edge, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef=0.3, neg_weight=1, generation_weight=0.2):
    n_pivot = len((modal_id[0]==2).nonzero())
    n_pivot = len((modal_id[0]==0).nonzero())//n_pivot
    # assert n_pivot==4
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==0]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    generation_weights = []
    for i in range(len(point_similarities)-1):
        generation_weights.append(generation_weight)
    generation_weights.append(1.0)
    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)*weight
        for (point_similarity, weight)
        in zip(point_similarities, generation_weights)]
    point_edge_loss_list = []
    point_loss = 0
    for i in range(len(total_edge_loss_generations_instance)):
        point_edge_loss = total_edge_loss_generations_instance[i]
        for j in range(3):
            selected_idxs = torch.nonzero(modal_id[0]==j)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            point_edge_loss_list.append(torch.mean(point_edge_loss[:,start:end, start:end]))
            point_loss += point_edge_loss_list[-1]

    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)*weight
        for (distribution_similarity, weight)
        in zip(distribution_similarities, generation_weights)]
    # for i in range(len(total_edge_loss_generations_distribution)):
    #     total_edge_loss_generations_distribution[i][ignore_mask<=0] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = 1
    total_edge_loss_generations = [
        distribution_loss_coeff * total_edge_loss_distribution
        for total_edge_loss_distribution in total_edge_loss_generations_distribution]
    
    # total_loss = 1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,:32])\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,32:64])+\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,64:])+\
    #     + dis_coef*point_loss/len(point_edge_loss_list)
    # total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0))*(all_label_in_edge.shape[1]/n_pivot) + dis_coef*point_loss/len(point_edge_loss_list)
    # total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0)) + dis_coef*point_loss/len(point_edge_loss_list)
    total_loss = point_loss/len(point_edge_loss_list)

    return total_loss

def compute_loss3(all_label_in_edge, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef=0.3, neg_weight=1, generation_weight=0.2):
    n_pivot = len((modal_id[0]==2).nonzero())
    n_pivot = len((modal_id[0]==0).nonzero())//n_pivot
    # assert n_pivot==4
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==0]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    generation_weights = []
    for i in range(len(point_similarities)-1):
        generation_weights.append(generation_weight)
    generation_weights.append(1.0)
    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)*weight
        for (point_similarity, weight)
        in zip(point_similarities, generation_weights)]
    point_edge_loss_list = []
    point_loss = 0
    for i in range(len(total_edge_loss_generations_instance)):
        point_edge_loss = total_edge_loss_generations_instance[i]
        for j in range(3):
            selected_idxs = torch.nonzero(modal_id[0]==j)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            point_edge_loss_list.append(torch.mean(point_edge_loss[:,start:end, start:end]))
            point_loss += point_edge_loss_list[-1]

    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)*weight
        for (distribution_similarity, weight)
        in zip(distribution_similarities, generation_weights)]
    # for i in range(len(total_edge_loss_generations_distribution)):
    #     total_edge_loss_generations_distribution[i][ignore_mask<=0] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = 1
    total_edge_loss_generations = [
        distribution_loss_coeff * total_edge_loss_distribution
        for total_edge_loss_distribution in total_edge_loss_generations_distribution]
    
    # total_loss = 1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,:32])\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,32:64])+\
    #     +1/3*torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:4,64:])+\
    #     + dis_coef*point_loss/len(point_edge_loss_list)
    # total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0))*(all_label_in_edge.shape[1]/n_pivot) + dis_coef*point_loss/len(point_edge_loss_list)
    total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0))

    return total_loss


def compute_loss(all_label_in_edge, point_similarities, distribution_similarities, dis_coef=0.1, ignore_mask=None, neg_weight=1):
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==0]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)
        for point_similarity
        in point_similarities]
    for i in range(len(total_edge_loss_generations_instance)):
        total_edge_loss_generations_instance[i][ignore_mask]=0
    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)
        for distribution_similarity
        in distribution_similarities]
    # for i in range(len(total_edge_loss_generations_distribution)):
    #     total_edge_loss_generations_distribution[i][ignore_mask] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = dis_coef
    total_edge_loss_generations = [
        total_edge_loss_instance + distribution_loss_coeff * total_edge_loss_distribution
        for (total_edge_loss_instance, total_edge_loss_distribution)
        in zip(total_edge_loss_generations_instance, total_edge_loss_generations_distribution)]
    
    total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0)[:,:8,:])
    return total_loss

def prepare_meta_info(labels, num_face, num_body, num_voice, device):
    bsz = len(labels)
    num = num_face + num_body + num_voice

    distribution_node = torch.zeros((bsz, num, num))+1/num
    distribution_node += torch.eye(num).unsqueeze(0).repeat(bsz, 1, 1)*(1-num)/num
    distribution_node = distribution_node.to(device)

    all_edges_labels = (labels.unsqueeze(1)==labels.unsqueeze(2)).float()
    ignore_mask = torch.zeros_like(all_edges_labels)
    ignore_mask[:,:num_face, num_face:] = 1
    ignore_mask[:,num_face:num_face+num_body,:num_face] = 1
    ignore_mask[:,num_face:num_face+num_body,num_face+num_body:] = 1
    ignore_mask[:,-num_voice:, :-num_voice] = 1
    ignore_mask = ignore_mask==1

    return distribution_node, all_edges_labels, ignore_mask

def prepare_meta_info2(labels, num_face, num_body, num_voice, associated_matrix, device):
    bsz = len(labels)
    num = num_face + num_body + num_voice

    # distribution_node = torch.zeros((bsz, num, num))+1/num
    distribution_node = torch.eye(num).unsqueeze(0).repeat(bsz, 1, 1) + associated_matrix
    distribution_node = distribution_node.to(device)

    all_edges_labels = (labels.unsqueeze(1)==labels.unsqueeze(2)).float()
    ignore_mask = torch.zeros_like(all_edges_labels)
    ignore_mask[:,:num_face, :num_face] = 1
    ignore_mask[:,num_face:num_face+num_body, num_face:num_face+num_body] = 1
    ignore_mask[:,-num_voice:, -num_voice] = 1

    ignore_mask = ignore_mask==0

    return distribution_node, all_edges_labels, ignore_mask


def cover_picture_selection(feat, threshold=0.6, max_num=2000):
    eps = 1e-5
    idxs = list(range(len(feat)))
    if len(idxs)>max_num:
        idxs = random.sample(idxs, max_num)
    feat = feat[idxs]
    sims = 1 - cdist(feat, feat, 'cosine')

    ds = np.where(sims>threshold, sims, 0)
    density = np.sum(ds, axis=1)

    ix, iy = (sims==1.0).nonzero()
    for i,j in zip(ix, iy):
        if i<j:
            density[j]=0
    density = density / (np.max(density) + eps)

    didxs = np.argsort(-density)

    rou = [0 for i in range(len(sims))]
    for i in range(len(sims)):
        d = 1 - np.min(sims[didxs[i]])
        for j in range(0, i):
            if density[didxs[j]] <= density[didxs[i]]:
                continue
            d = min(d, 1 - sims[didxs[i], didxs[j]])
        rou[didxs[i]]=d
    rou = np.array(rou) / (max(rou) + eps)

    priority = density * rou
    return priority, np.array(idxs, np.int)


def load_data(root, with_back=False):
    face_feats = np.load(osp.join(root, 'face_feats.npy'), allow_pickle=True)
    face_labels = np.loadtxt(osp.join(root, 'face_labels.txt'), dtype=np.str, delimiter='\n')
    face_track_ids = np.loadtxt(osp.join(root, 'face_track_ids.txt'), dtype=np.str)
    face_priority = np.load(osp.join(root, 'face_priority.npy'), allow_pickle=True)
    face_topk_track = np.loadtxt(osp.join(root, 'face_topk_track.txt'), dtype=np.int)
    try:
        face_frames = np.loadtxt(osp.join(root, 'face_frames.txt'), dtype=np.int)
    except:
        face_frames = None
    if with_back:
        body_feats = np.load(osp.join(root, 'bbody_feats.npy'), allow_pickle=True)
        body_labels = np.loadtxt(osp.join(root, 'bbody_labels.txt'), dtype=np.str, delimiter='\n')
        body_track_ids = np.loadtxt(osp.join(root, 'bbody_track_ids.txt'), dtype=np.str)
        body_priority = np.load(osp.join(root, 'bbody_priority.npy'), allow_pickle=True)
        body_topk_track = np.loadtxt(osp.join(root, 'bbody_topk_track.txt'), dtype=np.int)
        try:
            body_frames = np.loadtxt(osp.join(root, 'bbody_frames.txt'), dtype=np.int)
        except:
            body_frames = None
    else:
        body_feats = np.load(osp.join(root, 'body_feats.npy'), allow_pickle=True)
        body_labels = np.loadtxt(osp.join(root, 'body_labels.txt'), dtype=np.str, delimiter='\n')
        body_track_ids = np.loadtxt(osp.join(root, 'body_track_ids.txt'), dtype=np.str)
        body_priority = np.load(osp.join(root, 'body_priority.npy'), allow_pickle=True)
        body_topk_track = np.loadtxt(osp.join(root, 'body_topk_track.txt'), dtype=np.int)
        try:
            body_frames = np.loadtxt(osp.join(root, 'body_frames.txt'), dtype=np.int)
        except:
            body_frames = None

    voice_feats = np.load(osp.join(root, 'voice_feats.npy'))
    voice_labels = np.loadtxt(osp.join(root, 'voice_labels.txt'), dtype=np.str, delimiter='\n')
    voice_track_ids = np.loadtxt(osp.join(root, 'voice_track_ids.txt'), dtype=np.str)
    voice_topk_track = np.loadtxt(osp.join(root, 'voice_topk_track.txt'), dtype=np.int)

    face_infos = [face_feats, face_labels, face_track_ids, face_topk_track, face_frames, face_priority]
    for i in range(len(face_labels)):
        assert len(face_feats[i])==len(face_priority[i])
    body_infos = [body_feats, body_labels, body_track_ids, body_topk_track, body_frames, body_priority]
    voice_infos = [voice_feats, voice_labels, voice_track_ids, voice_topk_track]

    return face_infos, body_infos, voice_infos

def _indexing(labels, ignore=[-1]):
    d = {}
    for idx, la in enumerate(labels):
        if la in ignore:
            continue
        if la in d:
            d[la].append(idx)
        else:
            d[la]=[idx]
    return d


def load_data_list(root, dataset_list, folder='all', with_back=False, is_val=False):
    base_face = 0
    base_body = 0
    base_voice = 0
    face_feats_list, face_labels_list, face_track_ids_list, face_priority_list, face_topk_track_list, face_frames_list = [], [], [], [], [], []
    body_feats_list, body_labels_list, body_track_ids_list, body_priority_list, body_topk_track_list, body_frames_list = [], [], [], [], [], []
    voice_feats_list, voice_labels_list, voice_track_ids_list, voice_topk_track_list = [], [], [], []
    d_set = folder
    # if 'Friends' not in dataset_list and len(dataset_list)==5:
    #     dataset_list.append('Friends')
    if is_val:
        dataset_list = ['Friends']
    for pk in dataset_list:
        print(f'****************{pk}***************')
        eps = os.listdir(osp.join(root, pk))
        if pk=='Friends' and len(dataset_list)==5:
            eps = ['episode06','episode08','episode10','episode12','episode14',
                   'episode16','episode18','episode20','episode22','episode24']
        if is_val:
            eps = ['episode01','episode02','episode03','episode04','episode05']
        for ep in eps:
            print(f'>>>>>>>>>>>{ep}')
            if not is_val and len(dataset_list)==1 and dataset_list[0]=='Friends':
            # if pk=='Friends' and len(dataset_list)>1:
                if ep in ['episode01','episode02','episode03','episode04','episode05']:
                    print(f'skip {ep} for {pk}')
                    continue
            face_feats = np.load(osp.join(root, pk,ep, d_set, 'face_feats.npy'), allow_pickle=True)
            face_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_labels.txt'), dtype=np.str, delimiter='\n')
            face_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_track_ids.txt'), dtype=np.str)
            face_priority = np.load(osp.join(root, pk,ep, d_set, 'face_priority.npy'), allow_pickle=True)
            face_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_topk_track.txt'), dtype=np.int)
            face_frames = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_frames.txt'), dtype=np.int)

            body_feats = np.load(osp.join(root, pk,ep, d_set, 'body_feats.npy'), allow_pickle=True)
            body_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_labels.txt'), dtype=np.str, delimiter='\n')
            body_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_track_ids.txt'), dtype=np.str)
            body_priority = np.load(osp.join(root, pk,ep, d_set, 'body_priority.npy'), allow_pickle=True)
            body_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_topk_track.txt'), dtype=np.int)
            body_frames = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_frames.txt'), dtype=np.int)
            if with_back and (pk!='TBBT' or (ep!='s01e06'and ep!='s01e05')): 
                back_feats = np.load(osp.join(root, pk,ep, d_set, 'back_feats.npy'), allow_pickle=True)
                back_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_labels.txt'), dtype=np.str, delimiter='\n')
                back_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_track_ids.txt'), dtype=np.str)
                back_priority = np.load(osp.join(root, pk,ep, d_set, 'back_priority.npy'), allow_pickle=True)
                back_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_topk_body_track.txt'), dtype=np.int)
                back_frames = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_frames.txt'), dtype=np.int)

            voice_feats = np.load(osp.join(root, pk,ep, d_set, 'voice_feats.npy'), allow_pickle=True)
            voice_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'voice_labels.txt'), dtype=np.str, delimiter='\n')
            voice_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'voice_track_ids.txt'), dtype=np.str)
            voice_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'voice_topk_track.txt'), dtype=np.int)

            face_topks = []
            body_topks = []
            voice_topks = []
            face_ids = []
            body_ids = []
            voice_ids = []
            for face_topk in face_topk_track:
                face_topks.append(face_topk+base_face)

            for body_topk in body_topk_track:
                body_topks.append(body_topk+base_body)

            for voice_topk in voice_topk_track:
                voice_topks.append(voice_topk+base_voice)
            
            for face_id in face_track_ids:
                face_ids.append([pk+'_'+face_id[0],face_id[1]])

            for body_id in body_track_ids:
                body_ids.append([pk+'_'+body_id[0],body_id[1]])
            if with_back and (pk!='TBBT' or (ep!='s01e06'and ep!='s01e05')):
                for back_id in back_track_ids:
                    body_ids.append([pk+'_'+back_id[0],back_id[1]])
                for back_topk in back_topk_track:
                    a = back_topk+base_body
                    a = np.array([len(body_topks)]+a.tolist()[:-1], dtype=np.int)
                    body_topks.append(a)

            for voice_id in voice_track_ids:
                voice_ids.append([pk+'_'+voice_id[0],voice_id[1]])

            base_face += len(face_topk_track)
            base_body += len(body_ids)
            base_voice += len(voice_topk_track)
            print(base_face, base_body, base_voice)

            face_feats_list.append(face_feats)
            face_labels_list.append(face_labels)
            face_track_ids_list.append(face_ids)
            face_priority_list.append(face_priority)
            face_topk_track_list.append(face_topks)
            face_frames_list.append(face_frames) 

            body_feats_list.append(body_feats)
            body_labels_list.append(body_labels)
            body_track_ids_list.append(body_ids)
            body_priority_list.append(body_priority)
            body_topk_track_list.append(body_topks)
            body_frames_list.append(body_frames)
            if with_back and (pk!='TBBT' or (ep!='s01e06'and ep!='s01e05')):
                body_feats_list.append(back_feats)
                body_labels_list.append(back_labels)
                body_priority_list.append(back_priority)
                body_frames_list.append(back_frames)

            voice_feats_list.append(voice_feats)
            voice_labels_list.append(voice_labels)
            voice_track_ids_list.append(voice_ids)
            voice_topk_track_list.append(voice_topks)

    face_feats, face_labels, face_track_ids, face_topk_track, face_priority, face_frames = \
        cat(face_feats_list), cat(face_labels_list), cat(face_track_ids_list), cat(face_topk_track_list), cat(face_priority_list), cat(face_frames_list)
    body_feats, body_labels, body_track_ids, body_topk_track, body_priority, body_frames = \
        cat(body_feats_list), cat(body_labels_list), cat(body_track_ids_list), cat(body_topk_track_list), cat(body_priority_list), cat(body_frames_list)
    voice_feats, voice_labels, voice_track_ids, voice_topk_track = \
        cat(voice_feats_list), cat(voice_labels_list), cat(voice_track_ids_list), cat(voice_topk_track_list)

    face_infos = [face_feats, face_labels, face_track_ids, face_topk_track, face_frames, face_priority]
    for i in range(len(face_labels)):
        assert len(face_feats[i])==len(face_priority[i])
    body_infos = [body_feats, body_labels, body_track_ids, body_topk_track, body_frames, body_priority]
    voice_infos = [voice_feats, voice_labels, voice_track_ids, voice_topk_track]


    return face_infos, body_infos, voice_infos


def load_data_list_shuffle(root, dataset_list, folder='all', with_back=False, is_val=False, shuffle_rate=0.1):
    base_face = 0
    base_body = 0
    base_voice = 0
    face_feats_list, face_labels_list, face_track_ids_list, face_priority_list, face_topk_track_list, face_frames_list = [], [], [], [], [], []
    body_feats_list, body_labels_list, body_track_ids_list, body_priority_list, body_topk_track_list, body_frames_list = [], [], [], [], [], []
    voice_feats_list, voice_labels_list, voice_track_ids_list, voice_topk_track_list = [], [], [], []
    d_set = folder
    # if 'Friends' not in dataset_list and len(dataset_list)==5:
    #     dataset_list.append('Friends')
    if is_val:
        dataset_list = ['Friends']
    for pk in dataset_list:
        print(f'****************{pk}***************')
        eps = os.listdir(osp.join(root, pk))
        if pk=='Friends' and len(dataset_list)==5:
            eps = ['episode06','episode08','episode10','episode12','episode14',
                   'episode16','episode18','episode20','episode22','episode24']
        if is_val:
            eps = ['episode01','episode02','episode03','episode04','episode05']
        for ep in eps:
            print(f'>>>>>>>>>>>{ep}')
            if not is_val and len(dataset_list)==1 and dataset_list[0]=='Friends':
            # if pk=='Friends' and len(dataset_list)>1:
                if ep in ['episode01','episode02','episode03','episode04','episode05']:
                    print(f'skip {ep} for {pk}')
                    continue
            face_feats = np.load(osp.join(root, pk,ep, d_set, 'face_feats.npy'), allow_pickle=True)
            face_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_labels.txt'), dtype=np.str, delimiter='\n')
            face_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_track_ids.txt'), dtype=np.str)
            face_priority = np.load(osp.join(root, pk,ep, d_set, 'face_priority.npy'), allow_pickle=True)
            face_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_topk_track.txt'), dtype=np.int)
            face_frames = np.loadtxt(osp.join(root, pk,ep, d_set, 'face_frames.txt'), dtype=np.int)

            body_feats = np.load(osp.join(root, pk,ep, d_set, 'body_feats.npy'), allow_pickle=True)
            body_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_labels.txt'), dtype=np.str, delimiter='\n')
            body_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_track_ids.txt'), dtype=np.str)
            body_priority = np.load(osp.join(root, pk,ep, d_set, 'body_priority.npy'), allow_pickle=True)
            body_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_topk_track.txt'), dtype=np.int)
            body_frames = np.loadtxt(osp.join(root, pk,ep, d_set, 'body_frames.txt'), dtype=np.int)
            if with_back and (pk!='TBBT' or (ep!='s01e06'and ep!='s01e05')): # episodes没有back
                back_feats = np.load(osp.join(root, pk,ep, d_set, 'back_feats.npy'), allow_pickle=True)
                back_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_labels.txt'), dtype=np.str, delimiter='\n')
                back_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_track_ids.txt'), dtype=np.str)
                back_priority = np.load(osp.join(root, pk,ep, d_set, 'back_priority.npy'), allow_pickle=True)
                back_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_topk_body_track.txt'), dtype=np.int)
                back_frames = np.loadtxt(osp.join(root, pk,ep, d_set, 'back_frames.txt'), dtype=np.int)

            voice_feats = np.load(osp.join(root, pk,ep, d_set, 'voice_feats.npy'), allow_pickle=True)
            voice_labels = np.loadtxt(osp.join(root, pk,ep, d_set, 'voice_labels.txt'), dtype=np.str, delimiter='\n')
            voice_track_ids = np.loadtxt(osp.join(root, pk,ep, d_set, 'voice_track_ids.txt'), dtype=np.str)
            voice_topk_track = np.loadtxt(osp.join(root, pk,ep, d_set, 'voice_topk_track.txt'), dtype=np.int)

            face_topks = []
            body_topks = []
            voice_topks = []
            face_ids = []
            body_ids = []
            voice_ids = []
            for face_topk in face_topk_track:
                face_topks.append(face_topk+base_face)

            for body_topk in body_topk_track:
                body_topks.append(body_topk+base_body)

            for voice_topk in voice_topk_track:
                voice_topks.append(voice_topk+base_voice)
            
            for face_id in face_track_ids:
                face_ids.append([pk+'_'+face_id[0],face_id[1]])

            for body_id in body_track_ids:
                body_ids.append([pk+'_'+body_id[0],body_id[1]])
            if with_back and (pk!='TBBT' or (ep!='s01e06'and ep!='s01e05')):
                for back_id in back_track_ids:
                    body_ids.append([pk+'_'+back_id[0],back_id[1]])
                for back_topk in back_topk_track:
                    a = back_topk+base_body
                    a = np.array([len(body_topks)]+a.tolist()[:-1], dtype=np.int)
                    body_topks.append(a)

            for voice_id in voice_track_ids:
                voice_ids.append([pk+'_'+voice_id[0],voice_id[1]])

            base_face += len(face_topk_track)
            base_body += len(body_ids)
            base_voice += len(voice_topk_track)
            print(base_face, base_body, base_voice)

            face_feats_list.append(face_feats)
            face_labels_list.append(face_labels)
            face_track_ids_list.append(face_ids)
            face_priority_list.append(face_priority)
            face_topk_track_list.append(face_topks)
            face_frames_list.append(face_frames) 

            body_feats_list.append(body_feats)
            body_labels_list.append(body_labels)
            body_track_ids_list.append(body_ids)
            body_priority_list.append(body_priority)
            body_topk_track_list.append(body_topks)
            body_frames_list.append(body_frames)
            if with_back and (pk!='TBBT' or (ep!='s01e06'and ep!='s01e05')):
                body_feats_list.append(back_feats)
                body_labels_list.append(back_labels)
                body_priority_list.append(back_priority)
                body_frames_list.append(back_frames)

            voice_feats_list.append(voice_feats)
            voice_labels_list.append(voice_labels)
            voice_track_ids_list.append(voice_ids)
            voice_topk_track_list.append(voice_topks)

    face_feats, face_labels, face_track_ids, face_topk_track, face_priority, face_frames = \
        cat(face_feats_list), cat(face_labels_list), cat(face_track_ids_list), cat(face_topk_track_list), cat(face_priority_list), cat(face_frames_list)
    body_feats, body_labels, body_track_ids, body_topk_track, body_priority, body_frames = \
        cat(body_feats_list), cat(body_labels_list), cat(body_track_ids_list), cat(body_topk_track_list), cat(body_priority_list), cat(body_frames_list)
    voice_feats, voice_labels, voice_track_ids, voice_topk_track = \
        cat(voice_feats_list), cat(voice_labels_list), cat(voice_track_ids_list), cat(voice_topk_track_list)

    face_infos = [face_feats, face_labels, face_track_ids, face_topk_track, face_frames, face_priority]
    for i in range(len(face_labels)):
        assert len(face_feats[i])==len(face_priority[i])

    # episodes = body_track_ids[:, 0]
    # id_mapping = {i:i for i in range(body_track_ids.shape[0])}
    # for ep in np.unique(episodes):
    #     ids = np.where(episodes == ep)[0]
    #     num_sf = int(len(ids) * shuffle_rate)
    #     origin_ids = random.sample(ids.tolist(), num_sf)
    #     import copy
    #     new_ids = copy.deepcopy(origin_ids)
    #     random.shuffle(new_ids)
    #     for origin_id, new_id in zip(origin_ids, new_ids):
    #         id_mapping[origin_id] = new_id
    # import json
    # with open(os.path.join('shuffle_id', dataset_list[0]+'_'+'{:.2f}'.format(shuffle_rate)+'.json'), "w") as f:
    #     json.dump(id_mapping, f)
    import json
    with open(os.path.join('shuffle_id', dataset_list[0]+'_'+'{:.2f}'.format(shuffle_rate)+'.json'), "r") as f:
        res = f.read()
        id_map = json.loads(res)
    int_id_map = {}
    for key, val in id_map.items():
        int_id_map.update({int(key):int(val)})
    assert len(int_id_map) == body_feats.shape[0]
    body_infos = [body_feats, body_labels, body_track_ids, body_topk_track, body_frames, body_priority]
    voice_infos = [voice_feats, voice_labels, voice_track_ids, voice_topk_track]

    return face_infos, body_infos, voice_infos, int_id_map

def norm(X):
    for ix,x in enumerate(X):
        X[ix]/=np.linalg.norm(x)
    return X

def plot_embedding(X,Y):
    x_min, x_max = np.min(X,0), np.max(X,0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10,10))
    for i in range(X.shape[0]):
        plt.text(X[i,0],X[i,1], str(Y[i]),
                color=plt.cm.Set1(Y[i]/10.),
                fontdict={'weight':'bold','size':12})
    plt.savefig('a.jpg')


EPS = np.finfo(float).eps


def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.
    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    precision : float
        B-cubed precision.
    recall : float
        B-cubed recall.
    f1 : float
        B-cubed F1.
    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

def edge_generator2(similarity, neighbor, threshold=0.6, MinPts=2):
    u, v = [], []
    for i in range(similarity.shape[0]):
        sim, nbr = similarity[i], neighbor[i]
        idx = (sim > threshold).nonzero()[0]
        if len(idx) < MinPts:
            continue
        for j in idx:
            if nbr[j] != i:
                u.append(i)
                v.append(nbr[j])
    return np.array([u, v]).T

def edge_generator3(similarity, neighbor, threshold=0.6, MinPts=2):
    u, v = [], []
    score = []
    for i in range(similarity.shape[0]):
        sim, nbr = similarity[i], neighbor[i]
        idx = (sim > threshold).nonzero()[0]
        if len(idx) < MinPts:
            continue
        for j in idx:
            if nbr[j] != i:
                u.append(i)
                v.append(nbr[j])
                score.append(sim[j])
    return np.array([u, v]).T, np.array(score)


def compute_loss_vox(all_label_in_edge, point_similarities, distribution_similarities, dis_coef=0.1,  ignore_mask=None, neg_weight=1):
    weight = torch.zeros_like(all_label_in_edge)+1
    weight[all_label_in_edge==1]=neg_weight
    if neg_weight==1:
        edge_loss = nn.BCELoss(reduction='none')
    else:
        edge_loss = nn.BCELoss(reduction='none', weight=weight)

    # Point Loss
    total_edge_loss_generations_instance = [
        edge_loss(point_similarity, all_label_in_edge)
        for point_similarity
        in point_similarities]
    for i in range(len(total_edge_loss_generations_instance)):
        total_edge_loss_generations_instance[i][ignore_mask]=0
    # import pdb; pdb.set_trace()
    # Distribution Loss
    total_edge_loss_generations_distribution = [
        edge_loss(distribution_similarity, all_label_in_edge)
        for distribution_similarity
        in distribution_similarities]
    for i in range(len(total_edge_loss_generations_distribution)):
        total_edge_loss_generations_distribution[i][ignore_mask] = 0

    # combine Point Loss and Distribution Loss
    distribution_loss_coeff = dis_coef
    total_edge_loss_generations = [
        total_edge_loss_instance + distribution_loss_coeff * total_edge_loss_distribution
        for (total_edge_loss_instance, total_edge_loss_distribution)
        in zip(total_edge_loss_generations_instance, total_edge_loss_generations_distribution)]

    total_loss = torch.mean(torch.cat(total_edge_loss_generations, 0)[:,0,:])
    return total_loss

def CPR_test(gts, pds):
    from scipy.optimize import linear_sum_assignment

    gt_dict = _indexing(gts)
    pd_dict = _indexing(pds)
    m,n= len(gt_dict), len(pd_dict)

    cost = np.zeros((m, max(m,n)))
    gks = list(gt_dict.keys())
    pks = list(pd_dict.keys())
    # print('generate cost matrix ...')
    for i in range(m):
        for j in range(n):
            cost[i,j] = len((set(gt_dict[gks[i]]))&set(pd_dict[pks[j]]))/len(gt_dict[gks[i]])
    cost = - cost
    row_ind,col_ind=linear_sum_assignment(cost)

    cps = []
    crs = []
    for i in range(m):
        if col_ind[i]>=n:
            cps.append(0)
            crs.append(0)
        else:
            cps.append(len((set(gt_dict[gks[i]])) & set(pd_dict[pks[col_ind[i]]]))/len(pd_dict[pks[col_ind[i]]]))
            crs.append(len((set(gt_dict[gks[i]])) & set(pd_dict[pks[col_ind[i]]]))/len(gt_dict[gks[i]]))

    cps = np.array(cps)
    crs = np.array(crs)
    CP = np.mean(cps)
    CR = np.mean(crs)
    CF = 2*CR*CP/(CR+CP)
    # print(f'CP={CP}, CR={CR}, CF={2*CR*CP/(CR+CP)}')
    return CP, CR, CF

def edge_to_connected_graph(edges, data_num):
    father = np.arange(data_num)
    for i in range(edges.shape[0]):
        u, v = edges[i]
        fa_u = get_father(father, u)
        fa_v = get_father(father, v)
        father[fa_u] = fa_v

    for i in range(data_num):
        get_father(father, i)
    return father


def get_father(father, u):
    idx = []
    while u != father[u]:
        idx.append(u)
        u = father[u]
    for i in idx:
        father[i] = u
    return u
