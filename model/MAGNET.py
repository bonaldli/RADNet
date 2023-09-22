import torch.nn as nn
import torch.nn.functional as F
import torch

def cosine_similarity(f_x, f_y=None):
    f_xx_normalized = F.normalize(f_x, p=2, dim=-1)
    if f_y is None:
        f_yy_normalized = f_xx_normalized
    else:
        f_yy_normalized = F.normalize(f_y, p=2, dim=-1)
    f_yy_normalized_transpose = f_yy_normalized.transpose(1, 2)
    cosine_dis = torch.bmm(f_xx_normalized, f_yy_normalized_transpose)
    return cosine_dis

class alpha(nn.Module):
    def __init__(self, device, in_c):
        super(alpha, self).__init__()
        self.device = device
        self.in_c = in_c
        # self.alpha_generation = nn.ModuleList([nn.Sequential(nn.Linear(self.in_c, 2),
        #                     nn.Sigmoid()) for _ in range(3)])
        self.alpha_generation = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=1, bias=True),
                                             nn.Sigmoid()])
    
    def forward(self, feature_sims, ass_sims):
        feature_sims = torch.cat([feature_sims, ass_sims], dim=-1)
        alpha = self.alpha_generation(feature_sims)
        return alpha

class MultiModal_FeatureSimilarity(nn.Module):
    def __init__(self, device, num_modal, num_sample, in_c_list, base_c_list, dropout=0.0):
        super(MultiModal_FeatureSimilarity, self).__init__()
        self.num_modal = num_modal
        self.device = device
        self.featuresims = nn.ModuleList([FeatureSimilarity(self.device, in_c_list[i], base_c_list[i], dropout=dropout) for i in range(len(in_c_list))])
        # self.alpha_module = alpha(self.device, num_sample)
    
    def forward(self, vp_last_gen, associated_matrix, distance_metric='l1', modal_id=None):
        ep_ij = torch.zeros((modal_id.shape[0], modal_id.shape[1], modal_id.shape[1])).to(self.device)
        node_similarity = torch.zeros_like(ep_ij).to(self.device)
        # node_alpha = torch.zeros_like(ep_ij).to(self.device)
        for i in range(self.num_modal):
            selected_idxs = torch.nonzero(modal_id[0]==i)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            single_vp = vp_last_gen[:,start:end,:]
            ep_ij[:,start:end, start:end], node_similarity[:,start:end, start:end] = self.featuresims[i](single_vp,distance_metric)
        # node_alpha = self.alpha_module(node_similarity, associated_matrix)
        return ep_ij, node_similarity

class FeatureSimilarity(nn.Module):
    def __init__(self, device, in_c, base_c, dropout=0.0):
        super(FeatureSimilarity, self).__init__()
        self.device = device
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        

    def forward(self, vp_last_gen, distance_metric='l1'):
        # feature_alpha = torch.sigmoid(self.feature_alpha_generation(vp_last_gen))
        # vp_i = vp_last_gen.unsqueeze(2)
        # vp_j = torch.transpose(vp_i, 1, 2)
        # if distance_metric == 'l2':
        #     vp_similarity = (vp_i - vp_j)**2
        # elif distance_metric == 'l1':
        #     vp_similarity = torch.abs(vp_i - vp_j)
        # trans_similarity = torch.transpose(vp_similarity, 1, 3)
        # ep_ij = self.feature_sim_transform(vp_similarity)
        # ep_ij = torch.sigmoid(ep_ij).squeeze(-1)
        ep_ij = torch.relu(cosine_similarity(vp_last_gen)-1e-4)
       
        node_similarity = ep_ij
        # normalization
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1)
        diagonal_mask = diagonal_mask.to(self.device)

        ep_ij= ep_ij * diagonal_mask
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(self.device)
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        return ep_ij, node_similarity

def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.solve(eye, b_mat)
    return b_inv

def distribution_propagetion2(feature_edge, distribution_node, associated_matrix, alpha, device):
    associated_matrix = alpha*(feature_edge*0.8+associated_matrix*0.2)
    eye = torch.eye(feature_edge.shape[1]).repeat(feature_edge.shape[0],1 ,1).to(device)
    # b = [t.squeeze(0).inverse() for t in (eye - associated_matrix).chunk(feature_edge.shape[0], 0)]
    # c = torch.stack(b)
    c = b_inv(eye - associated_matrix)
    distribution_node = torch.bmm(c, distribution_node)
    # distribution_node = alpha*distribution_node + (1-alpha)*torch.bmm(associated_matrix, distribution_node)
    return distribution_node

def distribution_propagetion(feature_edge, distribution_node, associated_matrix, alpha, num_face, num_body, num_voice):
    coef = alpha.unsqueeze(-1).repeat(1,1,1,num_face)
    coef = coef.view(coef.shape[0], coef.shape[1], -1)[:,:,:(num_face+num_body+num_voice)]
    coef = coef*associated_matrix
    distribution_node = torch.bmm(feature_edge, distribution_node)
    distribution_node = (1-torch.sum(coef, dim=-1, keepdim=True))*distribution_node + torch.bmm(coef, distribution_node)
    return distribution_node

class F2DAgg(nn.Module):
    def __init__(self, in_c, out_c, device):
        super(F2DAgg, self).__init__()
        # add the fc layer
        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                             nn.LeakyReLU()])
        # self.edge_fusion = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
        #                                      nn.ReLU()])
        self.out_c = out_c
        self.device = device
        self.alpha_module = alpha(device, in_c)

    def forward(self, feature_edge, distribution_node, associated_matrix, num_face, num_body, num_voice):
        # associated_edge = torch.zeros_like(associated_matrix)
        # associated_edge  = associated_matrix + torch.eye(feature_edge.shape[1]).repeat(feature_edge.shape[0],1,1).float().to(self.device)
        associated_edge = associated_matrix.clone()
        associated_edge[associated_edge>0] = 1
        # alpha = ((associated_edge*feature_edge)>0).float() 
        # alpha = torch.sum(alpha, dim=-1, keepdim=True)
        # alpha = 1 - torch.sum(alpha, dim=1, keepdim=True)/(feature_edge.shape[1]*feature_edge.shape[1])
        # alpha = self.alpha_module(feature_edge, associated_matrix)
        D = torch.sum(feature_edge, dim=-1)
        D_12 = torch.pow(D, -0.5).unsqueeze(-1)
        D_12 = torch.eye(feature_edge.shape[1]).repeat(feature_edge.shape[0],1,1).float().to(self.device) * D_12
        S = torch.bmm(torch.bmm(D_12, feature_edge), D_12)
        
        # alpha = 0.3
        # import numpy as np
        # np.save('S.npy', S.cpu().detach().numpy())
        # np.save('associated_edge.npy', associated_edge.cpu().detach().numpy())
        # np.save('feature_edge.npy', feature_edge.cpu().detach().numpy())
        C = torch.bmm(torch.bmm(S, associated_edge), S.permute(0, 2, 1))
        alpha = self.alpha_module(feature_edge, C)
        feature_edge = alpha*C + (1-alpha)*feature_edge
        # np.save('feature_edge2.npy', feature_edge.cpu().detach().numpy())
        # feature_edge = F.normalize(feature_edge, p=1, dim=-1)

        # feature_edge = self.edge_fusion(torch.cat([feature_edge, associated_edge], dim=2))
        distribution_node = torch.cat([feature_edge, distribution_node], dim=2)
        # distribution_node = distribution_node.view(meta_batch*num_sample, -1)
        distribution_node = self.p2d_transform(distribution_node)
        # distribution_node = distribution_node.view(meta_batch, num_sample, -1)
        return distribution_node

class DistributionSimilarity(nn.Module):
    def __init__(self, device, in_c, base_c, dropout=0.0):
        super(DistributionSimilarity, self).__init__()
        self.device = device
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout

        self.feature_sim_transform = nn.Sequential(nn.Linear(self.in_c, 2*self.base_c), nn.PReLU(), nn.Dropout(p=self.dropout), 
                                nn.Linear(2*self.base_c, 1))

    def forward(self, vd_curr_gen, distance_metric='l1'):
        vd_i = vd_curr_gen.unsqueeze(2)
        vd_j = torch.transpose(vd_i, 1, 2)
        if distance_metric == 'l2':
            vd_similarity = (vd_i - vd_j)**2
        elif distance_metric == 'l1':
            vd_similarity = torch.abs(vd_i - vd_j)
        # trans_similarity = torch.transpose(vd_similarity, 1, 3)
        ed_ij = torch.sigmoid(self.feature_sim_transform(vd_similarity))
        ed_ij = ed_ij.squeeze(-1)
        dis_similarity = ed_ij

        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(self.device)
        ed_ij = ed_ij * diagonal_mask
        diagonal_reverse_mask = torch.eye(vd_curr_gen.size(1)).unsqueeze(0).to(self.device)
        ed_ij += (diagonal_reverse_mask + 1e-6)
        ed_ij /= torch.sum(ed_ij, dim=2).unsqueeze(-1)

        return ed_ij, dis_similarity


class D2FAgg(nn.Module):
    def __init__(self, device, in_c, base_c, dropout=0.0, last_generation=False):
        super(D2FAgg, self).__init__()
        self.device = device
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        self.last_generation = last_generation

        self.feature_node_transform = nn.ModuleList([nn.Linear(self.in_c, self.base_c) for _ in range(3)])
        self.raw_node_transform = nn.ModuleList([nn.Linear(self.in_c, self.base_c) for _ in range(3)])
        self.node_beta_layer = nn.ModuleList([nn.Sequential(nn.Linear(3*self.in_c, 1),
                            nn.Sigmoid()) for _ in range(3)])
        self.layer_norm = nn.ModuleList(nn.LayerNorm(self.base_c) for _ in range(3))

    def forward(self, distribution_edge, feature_node, modal_id):
        # get size
        meta_batch = feature_node.size(0)
        num_sample = feature_node.size(1)
        
        node_feat = []
        for i in range(3):
            selected_idxs = torch.nonzero(modal_id[0]==i)
            start = selected_idxs[0]
            end = selected_idxs[-1]+1
            single_de = distribution_edge[:,start:end,start:end]

            # get eye matrix (batch_size x node_size x node_size)
            diag_mask = 1.0 - torch.eye(single_de.size(1)).unsqueeze(0).repeat(meta_batch, 1, 1).to(self.device)

            # set diagonal as zero and normalize
            edge_feat = F.normalize(single_de * diag_mask, p=1, dim=-1)

            # compute attention and aggregate
            aggr_feat = self.raw_node_transform[i](torch.bmm(edge_feat, feature_node[:,start:end,:]))
    
            feature_i = self.feature_node_transform[i](feature_node[:,start:end,:])

            node_beta = self.node_beta_layer[i](torch.cat([feature_i, aggr_feat, feature_i-aggr_feat], dim=-1))

            # n_feat = feature_node[:,start:end,:]*node_beta + aggr_feat*(1-node_beta)
            # non-linear transform
            trans_feats = torch.relu(self.layer_norm[i](feature_i*node_beta + aggr_feat*(1-node_beta)))
            node_feat.append(trans_feats)
        node_feat = torch.cat(node_feat, dim=1)
        return node_feat

class MAGNET(nn.Module):
    def __init__(self, device, num_generations, dropout, num_sample, loss_indicator, feature_metric, distribution_metric):
        super(MAGNET, self).__init__()
        self.device = device
        self.generation = num_generations
        self.dropout = dropout
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.feature_metric = feature_metric
        self.distribution_metric = distribution_metric
        self.voice_projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256), nn.ReLU())
        self.body_projection = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU())
        self.face_projection = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU())
        # node & edge update module can be formulated by yourselves
        # self.init_P_Sim = MultiModal_FeatureSimilarity(self.device, 3, in_c_list=[256, 256, 256], base_c_list=[256, 256, 256], dropout=dropout)
        # self.init_D_Sim = DistributionSimilarity(self.device, num_sample, num_sample, dropout=dropout)
        self.distribution2feature_generation = nn.ModuleList([D2FAgg(self.device, 256, 256, dropout=self.dropout if l < self.generation-1 else 0.0, last_generation=l==(self.generation-1)) for l in range(self.generation)])
        self.feature2distribution_generation = nn.ModuleList([F2DAgg(num_sample*2, num_sample, device) for l in range(self.generation)])
        self.feature_sim_generation = nn.ModuleList([MultiModal_FeatureSimilarity(self.device, 3,num_sample, in_c_list=[256, 256, 256], base_c_list=[256, 256, 256], dropout=dropout if l < self.generation-1 else 0.0)
                                                    for l in range(self.generation)])
        self.distribution_sim_generation = nn.ModuleList([DistributionSimilarity(self.device, num_sample, num_sample,
                                        dropout=self.dropout if l < self.generation-1 else 0.0) for l in range(self.generation)])

    def forward(self, face_feats, body_feats, voice_feats, distribution_node, modal_id, associated_matrix):
        feature_similarities = []
        distribution_similarities = []
        # node_similarities_l2 = []
        voice_feats = self.voice_projection(voice_feats)
        face_feats = self.face_projection(face_feats)
        body_feats = self.body_projection(body_feats)
        feature_node = torch.cat([face_feats, body_feats, voice_feats], dim=1)
        num_face = face_feats.shape[1]
        num_body = body_feats.shape[1]
        num_voice = voice_feats.shape[1]
        # distribution_edge, _ = self.init_D_Sim(distribution_node, self.feature_metric)
        # import pdb; pdb.set_trace()
        for l in range(self.generation):
            feature_edge, node_similarity = self.feature_sim_generation[l](feature_node, associated_matrix, self.feature_metric, modal_id)
            # distribution_node = distribution_propagetion(feature_edge, distribution_node, associated_matrix, node_alpha, num_face, num_body, num_voice)
            distribution_node = self.feature2distribution_generation[l](node_similarity, distribution_node, associated_matrix, num_face, num_body, num_voice)
            distribution_edge, distribution_similairty = self.distribution_sim_generation[l](distribution_node, self.distribution_metric)
            feature_node = self.distribution2feature_generation[l](distribution_edge, feature_node, modal_id)
            feature_similarities.append(node_similarity * self.loss_indicator[0])
            # node_similarities_l2.append(node_similarity_l2 * self.loss_indicator[1])
            distribution_similarities.append(distribution_similairty * self.loss_indicator[1])
        return feature_similarities, distribution_similarities


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
    from util import prepare_meta_info2, compute_loss
    import numpy as np
    device = torch.device("cpu")
    torch.autograd.set_detect_anomaly(True)
    print(device)
    model = MAGNET(device, 2, 0.1, 136, [1,1], 'l1', 'l1').to(device)
    face_feats = torch.rand(72, 64, 512).to(device)
    body_feats = torch.rand(72, 64, 512).to(device)
    voice_feats = torch.rand(72, 8, 1024).to(device)
    labels = torch.arange(0, 8, 1).unsqueeze(0).repeat(72, 17).to(device)
    modal_id = []
    modal_id += [0]*64 +[1]*64+[2]*8
    modal_id = np.array(modal_id, dtype=np.int)
    modal_id = torch.from_numpy(modal_id).unsqueeze(0).repeat(72, 1).to(device)
    associated_matrix = torch.ones((72, 136, 136)).to(device)
    
    print(labels.shape)
    distribution_node, all_edges_labels, ignore_mask = prepare_meta_info2(labels, 64, 64, 8, associated_matrix, device)
    feature_similarities, distribution_similarities = model(face_feats, body_feats, voice_feats, distribution_node, modal_id, associated_matrix)
    loss = compute_loss(all_edges_labels, feature_similarities, distribution_similarities, ignore_mask=ignore_mask)
    print(loss)
    model.train()
    loss.backward()
