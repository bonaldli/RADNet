import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import pickle

from dataset3 import *
from model import *
import os
from util import *
from tqdm import tqdm
from inference_neigh_mmGcn_all import inference, inference_model

import argparse, yaml
from easydict import EasyDict as edict


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main(cfg):
    torch.autograd.set_detect_anomaly(True)
    # torch.cuda.manual_seed_all(2021)
    seed_ = cfg.train_cfg.seed
    setup_seed(seed_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = cfg.train_cfg.batch_size
    num_generations = cfg.train_cfg.num_generations
    dropout = cfg.train_cfg.dropout
    n_epochs = cfg.train_cfg.n_epochs
    dis_coef = cfg.train_cfg.dis_coef
    model_name = cfg.train_cfg.model
    neg_weight = cfg.train_cfg.neg_weight
    dataset_name = cfg.train_cfg.dataset
    nface_track = cfg.train_cfg.nface_track
    pface_cap = cfg.train_cfg.pface_cap
    nvoice_track = cfg.train_cfg.nvoice_track
    generation_weight = cfg.train_cfg.generation_weight
    init_dis = cfg.train_cfg.init_dis
    data_list = cfg.train_cfg.data_list

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

    root = '/Data2/new_VPCD_Release1/processed_dataset'
    face_infos, body_infos, voice_infos = load_data_list(root, data_list, folder='all', with_back=cfg.train_cfg.with_back)
    train_data = VPCD(face_infos, body_infos, voice_infos, graph_sampler_config, init_dis=init_dis)
    # face_infos, body_infos, voice_infos = load_data_list(root, settings[dataset_name][1], folder='all', with_back=True)
    # val_data = VPCD(face_infos, body_infos, voice_infos, graph_sampler_config, is_training=False)

    trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    model = MAGNET(device, num_generations, dropout, num_face+num_body+num_voice, [1, 1], 'l1', 'l1')
    model = nn.DataParallel(model)
    model = model.cuda()

    
    LR = cfg.train_cfg.LR
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train_cfg.step_size, gamma = cfg.train_cfg.gamma)
    save_dir = './checkpoint/mm_clustering_sims2alpha_1_alphaModule/{}_{}_Ngeneration_{}_dropout_{}_nface_{}_pface_{}_nvoice_{}_discoef_{}_lr_{}_neg_weight_{}_generation_weight_{}_init_dis_{}'.format(dataset_name, model_name, num_generations, dropout, nface_track, pface_cap, nvoice_track, dis_coef, 1e-3, neg_weight, generation_weight, init_dis)
    print(f'save in {save_dir}')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(n_epochs):
        # setup_seed(seed_)
        # seed_ += 1
        for step, dat in enumerate(trainloader):
            # trainloader.dataset.seed += 1

            face_feats, body_feats, voice_feats, labels, modal_id, associated_matrix, distribution_node, all_edges_labels, ignore_mask  = dat

            face_feats = face_feats.to(device)
            body_feats = body_feats.to(device)
            voice_feats = voice_feats.to(device)
            labels = labels.to(device)
            all_edges_labels = all_edges_labels.to(device)
            # distribution_node = distribution_node.to(device)

            # _, all_edges_labels, ignore_mask = prepare_meta_info2(labels,num_face, num_body, num_voice, associated_matrix, device)

            optimizer.zero_grad()
            model.train()
            point_similarities, distribution_similarities = model(face_feats, body_feats, voice_feats, distribution_node, modal_id, associated_matrix)

            loss = compute_loss2(all_edges_labels, point_similarities, distribution_similarities, modal_id, ignore_mask, dis_coef,neg_weight=neg_weight, generation_weight=generation_weight)

            loss.backward()
            optimizer.step()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            if step % 10 == 0:
                print(
                    "[Epoch %d/%d] [lr %f] [Batch %d/%d] [loss: %f]"
                    % (epoch, n_epochs, lr, step, (len(train_data) / batch_size) , loss)
                )
        
        scheduler.step()
        torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        }, osp.join(save_dir, f'agg_{epoch}.pth'))
    # print(f'best epoch={best_epoch}, best loss={best_loss}')
    # inference(cfg.test_cfg, osp.join(save_dir, f'agg_{epoch}.pth'))
    inference_model(cfg.test_cfg, model)
    # inference(args, osp.join(save_dir, 'agg_best.pth'))


if __name__ =='__main__':
    from pprint import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='None', type=str)
    args = parser.parse_args() 
    assert args.cfg is not 'None'
    with open(args.cfg, 'r') as f:
        cfg = edict(yaml.full_load(f))
    pprint(cfg)                
    main(cfg)
