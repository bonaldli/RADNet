import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import pickle
from tqdm import tqdm

from dataset3 import *
from model import *
from util import *

import argparse
import sys
import os
from scipy.spatial.distance import cdist
import argparse, yaml
from easydict import EasyDict as edict

from evaluation.evaluation import test
from util import *
from sklearn.metrics import normalized_mutual_info_score

DATA_ROOT = 'new_VPCD_Release1/processed_dataset'

def test_wcp(gts, pds):
    N = len(gts)
    assert len(gts)==len(pds)
    pd_dict = _indexing(pds)
    pn = 0
    for k, v in pd_dict.items():
        nc = -1
        c = None
        gdict = _indexing(gts[v])
        for gk, gv in gdict.items():
            if len(gv)>nc:
                nc = len(gv)
        pn+=nc
    return pn/N

def inference(cfg, model_path):
    from pprint import pprint
    pprint(cfg)
    print('start inference')
    print(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = cfg.batch_size
    num_generations = cfg.num_generations
    dropout = cfg.dropout
    n_epochs = cfg.n_epochs
    dis_coef = cfg.dis_coef
    verbose = cfg.verbose
    batch_size = batch_size*2
    model_name = cfg.model
    neg_weight = cfg.neg_weight
    dataset_name = cfg.dataset
    nface_track = cfg.nface_track
    pface_cap = cfg.pface_cap
    nvoice_track = cfg.nvoice_track
    generation_weight = cfg.generation_weight
    init_dis = cfg.init_dis

    graph_sampler_config = {
        'num_face_track': nface_track,
        'num_body_track': nface_track,
        'num_voice_track': nvoice_track,
        'num_per_face_track':pface_cap,
        'num_per_body_track':pface_cap
    }
    num_face = graph_sampler_config['num_face_track']*graph_sampler_config['num_per_face_track']
    num_body = graph_sampler_config['num_body_track']*graph_sampler_config['num_per_body_track']
    num_voice = graph_sampler_config['num_voice_track']

    settings = {'ABFHS':[['About_Last_Night', 'Buffy','Friends', 'Hidden_Figures', 'Sherlock'], ['TBBT']],
            'ABFHT':[['About_Last_Night', 'Buffy','Friends', 'Hidden_Figures', 'TBBT'], ['Sherlock']],
            'ABFST':[['About_Last_Night', 'Buffy','Friends', 'Sherlock', 'TBBT'], ['Hidden_Figures']],
            'ABHST':[['About_Last_Night', 'Buffy','Hidden_Figures', 'Sherlock', 'TBBT'], ['Friends']],
            'AFHST':[['About_Last_Night', 'Friends','Hidden_Figures', 'Sherlock', 'TBBT'], ['Buffy']],
            'BFHST':[['Buffy','Friends', 'Hidden_Figures', 'Sherlock', 'TBBT'], ['About_Last_Night']]
            }

    face_infos, body_infos, voice_infos = load_data_list(DATA_ROOT, cfg.data_list, folder='all', with_back=cfg.with_back)
    val_data = VPCD(face_infos, body_infos, voice_infos, graph_sampler_config, init_dis=init_dis, is_training=False)

    valloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    model = MAGNET(device, num_generations, dropout, num_face+num_body+num_voice, [1, 1], 'l1', 'l1')
    model = nn.DataParallel(model)
    model = model.cuda()

    model.load_state_dict(torch.load(model_path)['model'])
    
    val_loss = 0
    model.eval()

    total_face_track = len(val_data.face_feats)
    total_body_track = len(val_data.body_feats)
    total_voice_track = len(val_data.voice_feats)
    
    val_face_neighbors = np.array(val_data.sub_graph)[:total_face_track,:val_data.num_face_track]
    val_face_sims = np.zeros_like(val_face_neighbors, dtype=np.float)
    val_body_neighbors = np.array(val_data.sub_graph)[total_face_track:total_face_track+total_body_track,val_data.num_face_track:val_data.num_face_track+val_data.num_body_track]
    val_body_sims = np.zeros_like(val_body_neighbors, dtype=np.float)
    val_voice_neighbors = np.array(val_data.sub_graph)[-total_voice_track:, -val_data.num_voice_track:]
    val_voice_sims = np.zeros_like(val_voice_neighbors, dtype=np.float)

    
    final_dis_sims = []
    with torch.no_grad():
        for step, dat in tqdm(enumerate(valloader), disable= not verbose):
            face_feats, body_feats, voice_feats, labels, modal_id, associated_matrix, distribution_node, all_edges_labels, ignore_mask  = dat

            face_feats = face_feats.to(device)
            body_feats = body_feats.to(device)
            voice_feats = voice_feats.to(device)
            labels = labels.to(device)
            all_edges_labels = all_edges_labels.to(device)
            # distribution_node = distribution_node.to(device)

            # _, all_edges_labels, ignore_mask = prepare_meta_info2(labels,num_face, num_body, num_voice, associated_matrix, device)

            point_similarities, distribution_similarities = model(face_feats, body_feats, voice_feats, distribution_node, modal_id, associated_matrix)
            loss = compute_loss2(all_edges_labels, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef,  neg_weight=neg_weight, generation_weight=generation_weight)
            # val_sims[step*batch_size:(step*batch_size+batch_size)] = distribution_similarities[-1][:,0,:].cpu().detach().numpy()
            final_dis_sims.append(distribution_similarities[-1].cpu().detach().numpy())
            val_loss += loss
    final_dis_sims =  np.concatenate(final_dis_sims, axis=0)

    ubidxs = []
    for ix in range(len(val_data.body_track_ids)):
        if val_data.body_track_ids[ix] not in val_data.face_trackid2idx:
            ubidxs.append(ix)
    face_sims = final_dis_sims[:total_face_track, :val_data.num_per_face_track, :num_face]
    face_sims = np.mean(face_sims, axis=1)
    for i in range(val_data.num_face_track):
        val_face_sims[:,i] = np.mean(face_sims[:, i*val_data.num_per_face_track:i*val_data.num_per_face_track+ val_data.num_per_face_track], axis=1)

    
    body_sims = final_dis_sims[total_face_track:total_face_track+total_body_track, num_face: num_face+val_data.num_per_body_track, num_face:num_face+num_body]
    body_sims = np.mean(body_sims, axis=1)
    for i in range(val_data.num_face_track):
        val_body_sims[:,i] = np.mean(body_sims[:, i*val_data.num_per_body_track:i*val_data.num_per_body_track+ val_data.num_per_body_track], axis=1)
    val_voice_sims = final_dis_sims[:, num_face+num_body, -num_voice:]
    val_loss /= len(valloader)
    # np.save(f'sims/val_face_sims_{dataset_name}_{num_generations}.npy', val_face_sims)
    # np.save(f'sims/val_sims_{dataset_name}_{num_generations}.npy', final_dis_sims)
    # np.save(f'sims/val_body_sims_{dataset_name}_{num_generations}.npy', val_body_sims)
    # np.save(f'sims/val_voice_sims_{dataset_name}_{num_generations}.npy', val_voice_sims)
    # np.save(f'sims/{dataset_name}_distribution_node.npy', all_distribution_nodes, allow_pickle=True)
    print('--> val_loss:',val_loss)
    best_fscore = -1
    best_test = None
    test_results = []
    ep_idxs = np.unique(val_data.face_frames[:,0]).tolist()
    
    thre = cfg.thre
    print('>>>>face')
    edges = edge_generator2(val_face_sims, val_face_neighbors, threshold=thre, MinPts=1)
    face_pred_labels = edge_to_connected_graph(edges, data_num=val_face_neighbors.shape[0])

    body_pred_labels = np.zeros(len(val_data.body_feats), dtype=np.int)-1
    for i in range(len(val_data.body_track_ids)):
        if val_data.body_track_ids[i] in val_data.face_trackid2idx:
            body_pred_labels[i] = face_pred_labels[val_data.face_trackid2idx[val_data.body_track_ids[i]]]

    na = 100000000
    for i in range(len(val_data.body_track_ids)):
        if val_data.body_track_ids[i] not in val_data.face_trackid2idx:
            back_sims = final_dis_sims[i+len(val_data.face_feats)][num_face: num_face+val_data.num_per_body_track, :-num_voice]
            back_sims = np.mean(back_sims, axis=0)
            bsims = []
            for ix in range(val_data.num_face_track):
                bsims.append(np.mean(back_sims[ix*val_data.num_per_face_track:(ix+1)*val_data.num_per_face_track]))
            for ix in range(val_data.num_face_track, val_data.num_face_track+val_data.num_body_track):
                bsims.append(np.mean(back_sims[ix*val_data.num_per_body_track:(ix+1)*val_data.num_per_body_track]))
            bsims = np.array(bsims)
            bsims[val_data.num_face_track] = 0
            bsidxs = np.argsort(-bsims)
            if bsims[bsidxs[0]]>thre:
                if bsidxs[0]>=val_data.num_face_track:
                    body_pred_labels[i] = body_pred_labels[val_data.sub_graph[i+total_face_track][bsidxs[0]]]
                else:
                    body_pred_labels[i] = face_pred_labels[val_data.sub_graph[i+total_face_track][bsidxs[0]]]
            else:
                body_pred_labels[i] = na
                na+=1
    pds = np.concatenate([face_pred_labels, body_pred_labels[ubidxs]])
    ep_idxs = np.unique(np.concatenate([val_data.face_frames[:,0], val_data.body_frames[:,0]], axis=0)).tolist()
    gts = np.concatenate([val_data.face_labels, val_data.body_labels[ubidxs]])
    frames = np.concatenate([val_data.face_frames[:,0], val_data.body_frames[:,0][ubidxs]], axis=0)
    res = []
    nmis = []
    wcps = []
    cprfs = []
    print('>>>>>>> all')
    for ep_idx in ep_idxs:
        sidxs = frames==ep_idx
        print(val_data.raw_face_labels[sidxs.nonzero()[0][0]].split('_')[1])
        a = test(gts[sidxs], pds[sidxs], verbose=verbose)
        nmis.append(normalized_mutual_info_score(gts[sidxs], pds[sidxs]))
        wcps.append(test_wcp(gts[sidxs], pds[sidxs]))
        cprfs.append(CPR_test(gts[sidxs], pds[sidxs]))
        a = a.split()
        res.append([float(a[1]), float(a[2]), float(a[3])])
    res = np.array(res)
    res = np.mean(res, axis=0)
    res[2] = 2*res[0]*res[1]/(res[0]+res[1])
    cprfs = np.array(cprfs)
    cprf = np.mean(cprfs, axis=0)
    nmi = np.mean(np.array(nmis))
    wcp = np.mean(np.array(wcps))
    print(cprf, wcp, nmi)


def inference_model(cfg, model):
    from pprint import pprint
    pprint(cfg)
    print('start inference')
    # print(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = cfg.batch_size
    num_generations = cfg.num_generations
    dropout = cfg.dropout
    n_epochs = cfg.n_epochs
    dis_coef = cfg.dis_coef
    verbose = cfg.verbose
    batch_size = batch_size*2
    model_name = cfg.model
    neg_weight = cfg.neg_weight
    dataset_name = cfg.dataset
    nface_track = cfg.nface_track
    pface_cap = cfg.pface_cap
    nvoice_track = cfg.nvoice_track
    generation_weight = cfg.generation_weight
    init_dis = cfg.init_dis

    graph_sampler_config = {
        'num_face_track': nface_track,
        'num_body_track': nface_track,
        'num_voice_track': nvoice_track,
        'num_per_face_track':pface_cap,
        'num_per_body_track':pface_cap
    }
    num_face = graph_sampler_config['num_face_track']*graph_sampler_config['num_per_face_track']
    num_body = graph_sampler_config['num_body_track']*graph_sampler_config['num_per_body_track']
    num_voice = graph_sampler_config['num_voice_track']

    settings = {'ABFHS':[['About_Last_Night', 'Buffy','Friends', 'Hidden_Figures', 'Sherlock'], ['TBBT']],
            'ABFHT':[['About_Last_Night', 'Buffy','Friends', 'Hidden_Figures', 'TBBT'], ['Sherlock']],
            'ABFST':[['About_Last_Night', 'Buffy','Friends', 'Sherlock', 'TBBT'], ['Hidden_Figures']],
            'ABHST':[['About_Last_Night', 'Buffy','Hidden_Figures', 'Sherlock', 'TBBT'], ['Friends']],
            'AFHST':[['About_Last_Night', 'Friends','Hidden_Figures', 'Sherlock', 'TBBT'], ['Buffy']],
            'BFHST':[['Buffy','Friends', 'Hidden_Figures', 'Sherlock', 'TBBT'], ['About_Last_Night']]
            }

    face_infos, body_infos, voice_infos = load_data_list(DATA_ROOT, cfg.data_list, folder='all', with_back=cfg.with_back)
    val_data = VPCD(face_infos, body_infos, voice_infos, graph_sampler_config, init_dis=init_dis, is_training=False)

    valloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    
    val_loss = 0
    model.eval()

    total_face_track = len(val_data.face_feats)
    total_body_track = len(val_data.body_feats)
    total_voice_track = len(val_data.voice_feats)
    
    val_face_neighbors = np.array(val_data.sub_graph)[:total_face_track,:val_data.num_face_track]
    val_face_sims = np.zeros_like(val_face_neighbors, dtype=np.float)
    val_body_neighbors = np.array(val_data.sub_graph)[total_face_track:total_face_track+total_body_track,val_data.num_face_track:val_data.num_face_track+val_data.num_body_track]
    val_body_sims = np.zeros_like(val_body_neighbors, dtype=np.float)
    val_voice_neighbors = np.array(val_data.sub_graph)[-total_voice_track:, -val_data.num_voice_track:]
    val_voice_sims = np.zeros_like(val_voice_neighbors, dtype=np.float)

    
    final_dis_sims = []
    with torch.no_grad():
        for step, dat in tqdm(enumerate(valloader), disable= not verbose):
            face_feats, body_feats, voice_feats, labels, modal_id, associated_matrix, distribution_node, all_edges_labels, ignore_mask  = dat

            face_feats = face_feats.to(device)
            body_feats = body_feats.to(device)
            voice_feats = voice_feats.to(device)
            labels = labels.to(device)
            all_edges_labels = all_edges_labels.to(device)
            # distribution_node = distribution_node.to(device)

            # _, all_edges_labels, ignore_mask = prepare_meta_info2(labels,num_face, num_body, num_voice, associated_matrix, device)

            point_similarities, distribution_similarities = model(face_feats, body_feats, voice_feats, distribution_node, modal_id, associated_matrix)
            loss = compute_loss2(all_edges_labels, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef,  neg_weight=neg_weight, generation_weight=generation_weight)
            # val_sims[step*batch_size:(step*batch_size+batch_size)] = distribution_similarities[-1][:,0,:].cpu().detach().numpy()
            final_dis_sims.append(distribution_similarities[-1].cpu().detach().numpy())
            val_loss += loss
    final_dis_sims =  np.concatenate(final_dis_sims, axis=0)
    ubidxs = []
    for ix in range(len(val_data.body_track_ids)):
        if val_data.body_track_ids[ix] not in val_data.face_trackid2idx:
            ubidxs.append(ix)
    face_sims = final_dis_sims[:total_face_track, :val_data.num_per_face_track, :num_face]
    face_sims = np.mean(face_sims, axis=1)
    for i in range(val_data.num_face_track):
        val_face_sims[:,i] = np.mean(face_sims[:, i*val_data.num_per_face_track:i*val_data.num_per_face_track+ val_data.num_per_face_track], axis=1)

    body_sims = final_dis_sims[total_face_track:total_face_track+total_body_track, num_face: num_face+val_data.num_per_body_track, num_face:num_face+num_body]
    body_sims = np.mean(body_sims, axis=1)
    for i in range(val_data.num_face_track):
        val_body_sims[:,i] = np.mean(body_sims[:, i*val_data.num_per_body_track:i*val_data.num_per_body_track+ val_data.num_per_body_track], axis=1)
    val_voice_sims = final_dis_sims[:, num_face+num_body, -num_voice:]
    val_loss /= len(valloader)
    # np.save(f'sims/val_face_sims_{dataset_name}_{num_generations}.npy', val_face_sims)
    # np.save(f'sims/val_sims_{dataset_name}_{num_generations}.npy', final_dis_sims)
    # np.save(f'sims/val_body_sims_{dataset_name}_{num_generations}.npy', val_body_sims)
    # np.save(f'sims/val_voice_sims_{dataset_name}_{num_generations}.npy', val_voice_sims)
    # np.save(f'sims/{dataset_name}_distribution_node.npy', all_distribution_nodes, allow_pickle=True)
    print('--> val_loss:',val_loss)
    best_fscore = -1
    best_test = None
    test_results = []
    ep_idxs = np.unique(val_data.face_frames[:,0]).tolist()
    thre = cfg.thre
    print('>>>>face')
    edges = edge_generator2(val_face_sims, val_face_neighbors, threshold=thre, MinPts=1)
    face_pred_labels = edge_to_connected_graph(edges, data_num=val_face_neighbors.shape[0])

    body_pred_labels = np.zeros(len(val_data.body_feats), dtype=np.int)-1
    for i in range(len(val_data.body_track_ids)):
        if val_data.body_track_ids[i] in val_data.face_trackid2idx:
            body_pred_labels[i] = face_pred_labels[val_data.face_trackid2idx[val_data.body_track_ids[i]]]

    na = 100000000
    for i in range(len(val_data.body_track_ids)):
        if val_data.body_track_ids[i] not in val_data.face_trackid2idx:
            back_sims = final_dis_sims[i+len(val_data.face_feats)][num_face: num_face+val_data.num_per_body_track, :-num_voice]
            back_sims = np.mean(back_sims, axis=0)
            bsims = []
            for ix in range(val_data.num_face_track):
                bsims.append(np.mean(back_sims[ix*val_data.num_per_face_track:(ix+1)*val_data.num_per_face_track]))
            for ix in range(val_data.num_face_track, val_data.num_face_track+val_data.num_body_track):
                bsims.append(np.mean(back_sims[ix*val_data.num_per_body_track:(ix+1)*val_data.num_per_body_track]))
            bsims = np.array(bsims)
            bsims[val_data.num_face_track] = 0
            bsidxs = np.argsort(-bsims)
            if bsims[bsidxs[0]]>thre:
                if bsidxs[0]>=val_data.num_face_track:
                    body_pred_labels[i] = body_pred_labels[val_data.sub_graph[i+total_face_track][bsidxs[0]]]
                else:
                    body_pred_labels[i] = face_pred_labels[val_data.sub_graph[i+total_face_track][bsidxs[0]]]
            else:
                body_pred_labels[i] = na
                na+=1
    pds = np.concatenate([face_pred_labels, body_pred_labels[ubidxs]])
    ep_idxs = np.unique(np.concatenate([val_data.face_frames[:,0], val_data.body_frames[:,0]], axis=0)).tolist()
    gts = np.concatenate([val_data.face_labels, val_data.body_labels[ubidxs]])
    frames = np.concatenate([val_data.face_frames[:,0], val_data.body_frames[:,0][ubidxs]], axis=0)
    res = []
    nmis = []
    wcps = []
    cprfs = []
    print('>>>>>>> all')
    for ep_idx in ep_idxs:
        sidxs = frames==ep_idx
        print(val_data.raw_face_labels[sidxs.nonzero()[0][0]].split('_')[1])
        a = test(gts[sidxs], pds[sidxs], verbose=verbose)
        nmis.append(normalized_mutual_info_score(gts[sidxs], pds[sidxs]))
        wcps.append(test_wcp(gts[sidxs], pds[sidxs]))
        cprfs.append(CPR_test(gts[sidxs], pds[sidxs]))
        a = a.split()
        res.append([float(a[1]), float(a[2]), float(a[3])])
    res = np.array(res)
    res = np.mean(res, axis=0)
    res[2] = 2*res[0]*res[1]/(res[0]+res[1])
    cprfs = np.array(cprfs)
    cprf = np.mean(cprfs, axis=0)
    nmi = np.mean(np.array(nmis))
    wcp = np.mean(np.array(wcps))
    print(cprf, wcp, nmi)