import torch.utils.data as data
import numpy as np
import torch
import pickle
import copy

from util import L2_normalize_numpy, _indexing
import random

class VPCD(data.Dataset):
    def __init__(self, face_infos, body_infos, voice_infos, graph_sampler_config, init_dis=0.8, is_training=True):
        self.face_feats, self.face_labels, self.face_track_ids, self.face_topk_track, self.face_frames, self.face_select_priority = face_infos
        self.body_feats, self.body_labels, self.body_track_ids, self.body_topk_track, self.body_frames, self.body_select_priority = body_infos
        self.voice_feats, self.voice_labels, self.voice_track_ids, self.voice_topk_track = voice_infos
        self.is_training = is_training
        self.init_dis = init_dis

        self.raw_face_labels = copy.deepcopy(self.face_labels)
        self.raw_body_labels = copy.deepcopy(self.body_labels)
        self.raw_voice_labels = copy.deepcopy(self.voice_labels)
        self.face_center_feat = []
        for fa in self.face_feats:
            self.face_center_feat.append(np.mean(fa, axis=0))
        self.face_center_feat = L2_normalize_numpy(np.array(self.face_center_feat))

        self.body_center_feat = []
        for fa in self.body_feats:
            self.body_center_feat.append(np.mean(fa, axis=0))
        self.body_center_feat = L2_normalize_numpy(np.array(self.body_center_feat))        

        self.seed = 0
        rmap = {}
        for idx, la in enumerate(self.face_labels):
            if la not in rmap:
                nla = len(rmap)
                rmap[la] = nla
            self.face_labels[idx] = rmap[la]
        
        for idx, la in enumerate(self.body_labels):
            if la not in rmap:
                nla = len(rmap)
                rmap[la] = nla
            self.body_labels[idx] = rmap[la]

        for idx, la in enumerate(self.voice_labels):
            if la not in rmap:
                nla = len(rmap)
                rmap[la] = nla
            self.voice_labels[idx] = rmap[la]
        self.face_labels = self.face_labels.astype(np.int)
        self.body_labels = self.body_labels.astype(np.int)
        self.voice_labels = self.voice_labels.astype(np.int)
        
        for idx, p in enumerate(self.face_select_priority):
            self.face_select_priority[idx] = p/np.sum(p)
        for idx, p in enumerate(self.body_select_priority):
            self.body_select_priority[idx] = p/np.sum(p)

        self.body_trackid2idx = {}
        self.face_trackid2idx = {}
        self.voice_trackid2idx = {}
        tmp = [it[0]+'/'+it[1] for it in self.face_track_ids]
        tmp =np.array(tmp, dtype=np.str)
        self.face_track_ids = tmp
        tmp = [it[0]+'/'+it[1] for it in self.body_track_ids]
        tmp =np.array(tmp, dtype=np.str)
        self.body_track_ids = tmp
        tmp = [it[0]+'/'+it[1] for it in self.voice_track_ids]
        tmp =np.array(tmp, dtype=np.str)
        self.voice_track_ids = tmp

        for idx, bid in enumerate(self.body_track_ids):
            self.body_trackid2idx[bid] = idx
        for idx, bid in enumerate(self.face_track_ids):
            self.face_trackid2idx[bid] = idx
        for idx, bid in enumerate(self.voice_track_ids):
            self.voice_trackid2idx[bid] = idx

        self.graph_sampler_config = graph_sampler_config
        self.num_face_track = self.graph_sampler_config['num_face_track']
        self.num_body_track = self.graph_sampler_config['num_body_track']
        self.num_voice_track = self.graph_sampler_config['num_voice_track']
        self.num_per_face_track = self.graph_sampler_config['num_per_face_track']
        self.num_per_body_track = self.graph_sampler_config['num_per_body_track']
        self.num_track_per_graph = self.num_face_track+self.num_body_track+self.num_voice_track

        
        self.sub_graph = []
        self.graph_index = []
        self._remove_overlap_track()
        if self.is_training:
            self._create_grapn_index()
        else:
            self._creat_sub_graph()
    
    def _isOverlap(self, query_idx, target_idx, frames):
        query_frame = frames[query_idx]
        target_frame = frames[target_idx]
        if query_frame[0]!=target_frame[0]:
            return False
        if max(query_frame[2], target_frame[2]) - min(query_frame[1], target_frame[1]) < query_frame[2]-query_frame[1]+target_frame[2]-target_frame[1]:
            return True
        return False

    def _remove_overlap_track(self):
        _face_topk_track = []
        cnt = 0
        for i in range(len(self.face_topk_track)):
            _face_topk_track.append([i])
            for j in range(1, self.face_topk_track.shape[1]):
                if not self._isOverlap(self.face_topk_track[i][j], _face_topk_track[-1][0], self.face_frames):
                    _face_topk_track[-1].append(self.face_topk_track[i][j])
                else:
                    cnt+=1
            _face_topk_track[-1] = np.array(_face_topk_track[-1])
        self.face_topk_track = _face_topk_track
        print(f'{cnt} removed from face topk')

        _body_topk_track = []
        cnt = 0
        for i in range(len(self.body_topk_track)):
            _body_topk_track.append([i])
            for j in range(1, self.body_topk_track.shape[1]):
                if not self._isOverlap(self.body_topk_track[i][j], _body_topk_track[-1][0], self.body_frames):
                    _body_topk_track[-1].append(self.body_topk_track[i][j])
                else:
                    cnt+=1
            _body_topk_track[-1] = np.array(_body_topk_track[-1])
        self.body_topk_track = _body_topk_track
        print(f'{cnt} removed from body topk')
    
    def _creat_sub_graph(self):
        print(f'start to creat sub graph ...', end=' ')
        for i in range(len(self.face_topk_track)):
            self.sub_graph.append(self.face_topk_track[i][:self.num_face_track].tolist())
            for fi in self.face_topk_track[i][:self.num_face_track]: 
                if self.face_track_ids[fi] in self.body_trackid2idx:
                    self.sub_graph[-1].append(self.body_trackid2idx[self.face_track_ids[fi]])

            if len(self.sub_graph[-1])==self.num_face_track:
                for fi in self.face_topk_track[i][self.num_face_track:]: 
                    if self.face_track_ids[fi] in self.body_trackid2idx:
                        self.sub_graph[-1].append(self.body_trackid2idx[self.face_track_ids[fi]])
                        break
            if len(self.sub_graph[-1])==self.num_face_track:
                
                
                self.sub_graph[-1].append(random.randint(0, len(self.body_feats)-1))

            num_left_body_track = self.num_face_track + self.num_body_track - len(self.sub_graph[-1])
            ib=0
            while num_left_body_track>0:
                select_body_track = self.body_topk_track[self.sub_graph[-1][self.num_face_track+ib]]
                for bid in select_body_track:
                    if bid in self.sub_graph[-1][self.num_face_track:]: 
                        continue
                    else:
                        self.sub_graph[-1].append(bid)
                        num_left_body_track -= 1
                    if num_left_body_track<=0:
                        break
                ib+=1

            for fi in self.face_topk_track[i][:self.num_face_track]: 
                if self.face_track_ids[fi] in self.voice_trackid2idx:
                    self.sub_graph[-1].append(self.voice_trackid2idx[self.face_track_ids[fi]])
            if len(self.sub_graph[-1])==(self.num_face_track+self.num_body_track):
                for fi in self.face_topk_track[i][self.num_face_track:]: 
                    if self.face_track_ids[fi] in self.voice_trackid2idx:
                        self.sub_graph[-1].append(self.voice_trackid2idx[self.face_track_ids[fi]])
                        break
            if len(self.sub_graph[-1])==(self.num_face_track+self.num_body_track):
                
                
                self.sub_graph[-1].append(random.randint(0, len(self.voice_feats)-1))

            num_left_voice_track = self.num_face_track + self.num_body_track + self.num_voice_track - len(self.sub_graph[-1])
            iv = 0
            while num_left_voice_track>0:
                select_voice_track = self.voice_topk_track[self.sub_graph[-1][self.num_face_track+self.num_body_track+iv]]
                for bid in select_voice_track:
                    if bid in self.sub_graph[-1][self.num_face_track+self.num_body_track:]: 
                        continue
                    else:
                        self.sub_graph[-1].append(bid)
                        num_left_voice_track -= 1
                    if num_left_voice_track<=0:
                        break
                iv+=1

            assert len(self.sub_graph[-1])==self.num_track_per_graph
            assert max(self.sub_graph[-1][:self.num_face_track])<len(self.face_feats)
            assert max(self.sub_graph[-1][self.num_face_track: self.num_face_track+self.num_body_track])<len(self.body_feats)
            assert max(self.sub_graph[-1][self.num_face_track+self.num_body_track:])<len(self.voice_feats)
        
        for i in range(len(self.body_topk_track)):
            if self.is_training and self.body_track_ids[i] in self.face_trackid2idx:
                continue

            self.sub_graph.append(self.body_topk_track[i][:self.num_body_track].tolist())
            add_face = []
            for fi in self.body_topk_track[i][: self.num_body_track]: 
                if self.body_track_ids[fi] in self.face_trackid2idx:
                    add_face.append(self.face_trackid2idx[self.body_track_ids[fi]])


            if len(add_face)==0:
                for fi in self.body_topk_track[i][self.num_body_track:]: 
                    if self.body_track_ids[fi] in self.face_trackid2idx:
                        add_face.append(self.face_trackid2idx[self.body_track_ids[fi]])
                        break

            if len(add_face)==0:
                
                
                add_face.append(random.randint(0, len(self.face_feats)-1))

            num_left_face_track = self.num_face_track - len(add_face)
            ia = 0
            while num_left_face_track>0:
                select_face_track = self.face_topk_track[add_face[0+ia]]
                for bid in select_face_track:
                    if bid in add_face: 
                        continue
                    else:
                        add_face.append(bid)
                        num_left_face_track -= 1
                    if num_left_face_track<=0:
                        break
                ia+=1
                
            self.sub_graph[-1] = add_face + self.sub_graph[-1]
            for fi in self.body_topk_track[i][:self.num_body_track]: 
                if self.body_track_ids[fi] in self.voice_trackid2idx:
                    self.sub_graph[-1].append(self.voice_trackid2idx[self.body_track_ids[fi]])
            if len(self.sub_graph[-1])==(self.num_face_track+self.num_body_track):
                for fi in self.body_topk_track[i][self.num_body_track:]: 
                    if self.body_track_ids[fi] in self.voice_trackid2idx:
                        self.sub_graph[-1].append(self.voice_trackid2idx[self.body_track_ids[fi]])
                        break
            if len(self.sub_graph[-1])==(self.num_face_track+self.num_body_track):
                
                
                self.sub_graph[-1].append(random.randint(0, len(self.voice_feats)-1))
            num_left_voice_track = self.num_face_track + self.num_body_track + self.num_voice_track - len(self.sub_graph[-1])
            iv = 0
            while num_left_voice_track>0:
                select_voice_track = self.voice_topk_track[self.sub_graph[-1][self.num_face_track+self.num_body_track+iv]]
                for bid in select_voice_track:
                    if bid in self.sub_graph[-1][self.num_face_track+self.num_body_track:]: 
                        continue
                    else:
                        self.sub_graph[-1].append(bid)
                        num_left_voice_track -= 1
                    if num_left_voice_track<=0:
                        break
                iv+=1
            assert len(self.sub_graph[-1])==self.num_track_per_graph
            assert max(self.sub_graph[-1][:self.num_face_track])<len(self.face_feats)
            assert max(self.sub_graph[-1][self.num_face_track: self.num_face_track+self.num_body_track])<len(self.body_feats)
            assert max(self.sub_graph[-1][self.num_face_track+self.num_body_track:])<len(self.voice_feats)
        
        for i in range(len(self.voice_topk_track)):
            if self.voice_track_ids[i] not in self.face_trackid2idx:
                continue
            self.sub_graph.append(self.voice_topk_track[i][:self.num_voice_track].tolist())
            add_face = []
            for fi in self.voice_topk_track[i][: self.num_voice_track]: 
                if self.voice_track_ids[fi] in self.face_trackid2idx:
                    add_face.append(self.face_trackid2idx[self.voice_track_ids[fi]])
                if len(add_face)>=self.num_face_track:
                    break

            if len(add_face)==0:
                for fi in self.voice_topk_track[i][self.num_voice_track:]: 
                    if self.voice_track_ids[fi] in self.face_trackid2idx:
                        add_face.append(self.face_trackid2idx[self.voice_track_ids[fi]])
                        break
            if len(add_face)==0:
                
                
                add_face.append(random.randint(0, len(self.face_feats)-1))

            num_left_face_track = self.num_face_track - len(add_face)
            if num_left_face_track>0:
                select_face_track = self.face_topk_track[add_face[0]]
                for bid in select_face_track:
                    if bid in add_face: 
                        continue
                    else:
                        add_face.append(bid)
                        num_left_face_track -= 1
                    if num_left_face_track<=0:
                        break
            add_body = []
            for fi in self.voice_topk_track[i][:self.num_voice_track]: 
                if self.voice_track_ids[fi] in self.body_trackid2idx:
                    add_body.append(self.body_trackid2idx[self.voice_track_ids[fi]])
                if len(add_body)>=self.num_body_track:
                    break
            if len(add_body)==0:
                for fi in self.voice_topk_track[i][self.num_voice_track:]: 
                    if self.voice_track_ids[fi] in self.body_trackid2idx:
                        add_body.append(self.body_trackid2idx[self.voice_track_ids[fi]])
                        break
            if len(add_body)==0:
                
                
                add_body.append(random.randint(0, len(self.body_feats)-1))
            num_left_body_track = self.num_body_track - len(add_body)
            if num_left_body_track>0:
                select_body_track = self.body_topk_track[add_body[0]]
                for bid in select_body_track:
                    if bid in add_body: 
                        continue
                    else:
                        add_body.append(bid)
                        num_left_body_track -= 1
                    if num_left_body_track<=0:
                        break
            self.sub_graph[-1] = add_face + add_body + self.sub_graph[-1]
            assert len(self.sub_graph[-1])==self.num_track_per_graph
            assert max(self.sub_graph[-1][:self.num_face_track])<len(self.face_feats)
            assert max(self.sub_graph[-1][self.num_face_track: self.num_face_track+self.num_body_track])<len(self.body_feats)
            assert max(self.sub_graph[-1][self.num_face_track+self.num_body_track:])<len(self.voice_feats)

        print(f'sub_graph num={len(self.sub_graph)}', end ='  ')
        print('done!')       

    def _creat_sub_graph_random(self, ix, i):
        sub_graph = []
        if ix==0:
            
            
            sub_graph.append([i])
            sub_graph[-1] += random.sample(self.face_topk_track[i].tolist()[1:], self.num_face_track-1)
            for fi in sub_graph[-1][:self.num_face_track]: 
                if self.face_track_ids[fi] in self.body_trackid2idx:
                   sub_graph[-1].append(self.body_trackid2idx[self.face_track_ids[fi]])

            if len(sub_graph[-1])==self.num_face_track:
                
                
                sub_graph[-1].append(random.randint(0, len(self.body_feats)-1))

            num_left_body_track = self.num_face_track + self.num_body_track - len(sub_graph[-1])
            ib = 0
            while num_left_body_track>0:
                select_body_track = self.body_topk_track[sub_graph[-1][self.num_face_track+ib]]
                for bid in select_body_track:
                    if bid in sub_graph[-1][self.num_face_track:]: 
                        continue
                    else:
                        sub_graph[-1].append(bid)
                        num_left_body_track -= 1
                    if num_left_body_track<=0:
                        break
                ib+=1

            for fi in self.face_topk_track[i][:self.num_face_track]: 
                if self.face_track_ids[fi] in self.voice_trackid2idx:
                    sub_graph[-1].append(self.voice_trackid2idx[self.face_track_ids[fi]])
            if len(sub_graph[-1])==(self.num_face_track+self.num_body_track):
                for fi in self.face_topk_track[i][self.num_face_track:]: 
                    if self.face_track_ids[fi] in self.voice_trackid2idx:
                        sub_graph[-1].append(self.voice_trackid2idx[self.face_track_ids[fi]])
                        break
            if len(sub_graph[-1])==(self.num_face_track+self.num_body_track):
                
                
                sub_graph[-1].append(random.randint(0, len(self.voice_feats)-1))

            num_left_voice_track = self.num_face_track + self.num_body_track + self.num_voice_track - len(sub_graph[-1])
            iv = 0
            while num_left_voice_track>0:
                select_voice_track = self.voice_topk_track[sub_graph[-1][self.num_face_track+self.num_body_track+iv]]
                for bid in select_voice_track:
                    if bid in sub_graph[-1][self.num_face_track+self.num_body_track:]: 
                        continue
                    else:
                        sub_graph[-1].append(bid)
                        num_left_voice_track -= 1
                    if num_left_voice_track<=0:
                        break
                iv+=1 

            assert len(sub_graph[-1])==self.num_track_per_graph
            assert max(sub_graph[-1][:self.num_face_track])<len(self.face_feats)
            assert max(sub_graph[-1][self.num_face_track: self.num_face_track+self.num_body_track])<len(self.body_feats)
            assert max(sub_graph[-1][self.num_face_track+self.num_body_track:])<len(self.voice_feats)
        
        elif ix==1:
            
            
            sub_graph.append([i]+random.sample(self.body_topk_track[i][:self.num_body_track].tolist(), self.num_body_track-1))
            add_face = []
            for fi in self.body_topk_track[i][: self.num_body_track]:
                if self.body_track_ids[fi] in self.face_trackid2idx:
                    add_face.append(self.face_trackid2idx[self.body_track_ids[fi]])

            if len(add_face)==0:
                for fi in self.body_topk_track[i][self.num_body_track:]: 
                    if self.body_track_ids[fi] in self.face_trackid2idx:
                        add_face.append(self.face_trackid2idx[self.body_track_ids[fi]])
                        break
            if len(add_face)==0:
                
                
                add_face.append(random.randint(0, len(self.face_feats)-1))

            num_left_face_track = self.num_face_track - len(add_face)
            ia = 0
            while num_left_face_track>0:
                select_face_track = self.face_topk_track[add_face[0+ia]]
                for bid in select_face_track:
                    if bid in add_face: 
                        continue
                    else:
                        add_face.append(bid)
                        num_left_face_track -= 1
                    if num_left_face_track<=0:
                        break
                ia+=1
                
            sub_graph[-1] = add_face + sub_graph[-1]
            for fi in self.body_topk_track[i][:self.num_body_track]: 
                if self.body_track_ids[fi] in self.voice_trackid2idx:
                    sub_graph[-1].append(self.voice_trackid2idx[self.body_track_ids[fi]])
            if len(sub_graph[-1])==(self.num_face_track+self.num_body_track):
                for fi in self.body_topk_track[i][self.num_body_track:]: 
                    if self.body_track_ids[fi] in self.voice_trackid2idx:
                        sub_graph[-1].append(self.voice_trackid2idx[self.body_track_ids[fi]])
                        break
            if len(sub_graph[-1])==(self.num_face_track+self.num_body_track):
                
                
                sub_graph[-1].append(random.randint(0, len(self.voice_feats)-1))
            num_left_voice_track = self.num_face_track + self.num_body_track + self.num_voice_track - len(sub_graph[-1])
            iv = 0
            while num_left_voice_track>0:
                select_voice_track = self.voice_topk_track[sub_graph[-1][self.num_face_track+self.num_body_track+iv]]
                for bid in select_voice_track:
                    if bid in sub_graph[-1][self.num_face_track+self.num_body_track:]: 
                        continue
                    else:
                        sub_graph[-1].append(bid)
                        num_left_voice_track -= 1
                    if num_left_voice_track<=0:
                        break
                iv+=1
            assert len(sub_graph[-1])==self.num_track_per_graph
            assert max(sub_graph[-1][:self.num_face_track])<len(self.face_feats)
            assert max(sub_graph[-1][self.num_face_track: self.num_face_track+self.num_body_track])<len(self.body_feats)
            assert max(sub_graph[-1][self.num_face_track+self.num_body_track:])<len(self.voice_feats)


        return sub_graph[-1]

    def _create_grapn_index(self):
        print(f'start to creat sub graph index ...', end=' ')
        for i in range(len(self.face_topk_track)):
            self.graph_index.append([0, i])

        for i in range(len(self.body_topk_track)):
            if self.body_track_ids[i] in self.face_trackid2idx:
                continue
            self.graph_index.append([1, i])
        
        print(f'sub_graph num={len(self.graph_index)}', end ='  ')
        print('done!')       

    
    def __getitem__(self, index):
        modal_id = [0]*(self.num_per_face_track*self.num_face_track) + [1]*(self.num_per_body_track*self.num_body_track) + [2]*self.num_voice_track
        modal_id = np.array(modal_id, dtype=np.int)

        if self.is_training:
            ix, i = self.graph_index[index]
            selected_track = self._creat_sub_graph_random(ix, i)
        else:
            selected_track = self.sub_graph[index]
        track_id = []
        for it in selected_track[:self.num_face_track]:
            track_id += [self.face_track_ids[it]]*self.num_per_face_track
        for it in selected_track[self.num_face_track:self.num_face_track+self.num_body_track]:
            track_id += [self.body_track_ids[it]]*self.num_per_body_track
        track_id += [self.voice_track_ids[it] for it in selected_track[self.num_face_track+self.num_body_track:]]
        track_id = np.array(track_id, dtype=np.str)
        association_matrix = (track_id.reshape(-1, 1)==track_id.reshape(1, -1)).astype(np.float32)

        distribution_node = association_matrix.copy() 
        distribution_node[distribution_node==1]=self.init_dis
        distribution_node[distribution_node==0]=1-self.init_dis 

        num_face = self.num_face_track*self.num_per_face_track
        num_body = self.num_face_track*self.num_per_body_track
        num_voice = self.num_voice_track
        association_matrix[:num_face, :num_face]=0
        association_matrix[num_face:num_face+num_body, num_face:num_face+num_body]=0
        association_matrix[-num_voice:,-num_voice] = 0
        association_matrix[:,:num_face] = association_matrix[:,:num_face]/self.num_per_face_track
        association_matrix[:,num_face:num_face+num_body] = association_matrix[:,num_face:num_face+num_body]/self.num_per_body_track

        face_labels = []
        face_feats = []
        face_center_feats = []
        for tid in selected_track[:self.num_face_track]:
            if self.is_training:
                ridxs = list(range(len(self.face_feats[tid])))
                
                sidxs = np.random.choice(ridxs, self.num_per_face_track, p=self.face_select_priority[tid])
            else:
                sidxs = np.argsort(-self.face_select_priority[tid])[:self.num_per_face_track] 
                while len(sidxs)<self.num_per_face_track: 
                    rn = self.num_per_face_track - len(sidxs)
                    sidxs = np.concatenate([sidxs, sidxs[:rn]])
            face_feats.append(self.face_feats[tid][sidxs])
            fa = np.mean(self.face_center_feat[self.face_topk_track[tid][1:17]], axis=0)
            fa = fa/np.linalg.norm(fa, ord=2)
            face_center_feats += [fa]*len(sidxs)
            face_labels += [self.face_labels[tid]]*self.num_per_face_track
        face_labels = np.array(face_labels, dtype=np.int)
        face_feats = np.concatenate(face_feats, axis=0).astype(np.float32)
        face_center_feats = np.array(face_center_feats).astype(np.float32)
        face_feats = np.concatenate([face_feats, face_center_feats], axis=1)
        
        body_labels = []
        body_feats = []
        body_center_feats = []
        for tid in selected_track[self.num_face_track: self.num_face_track+self.num_body_track]:
            if self.is_training:
                ridxs = list(range(len(self.body_feats[tid])))
                
                sidxs = np.random.choice(ridxs, self.num_per_body_track, p=self.body_select_priority[tid])
            else:
                sidxs = np.argsort(-self.body_select_priority[tid])[:self.num_per_body_track] 
                while len(sidxs)<self.num_per_body_track:
                    rn = self.num_per_body_track - len(sidxs)
                    sidxs = np.concatenate([sidxs, sidxs[:rn]])
            body_feats.append(self.body_feats[tid][sidxs])

            body_labels += [self.body_labels[tid]]*self.num_per_body_track
            fa = np.mean(self.body_center_feat[self.body_topk_track[tid][1:17]], axis=0)
            fa = fa/np.linalg.norm(fa, ord=2)
            body_center_feats += [fa]*len(sidxs)
        body_labels = np.array(body_labels, dtype=np.int)
        body_feats = np.concatenate(body_feats, axis=0).astype(np.float32)
        body_center_feats = np.array(body_center_feats, dtype=np.float32)
        body_feats = np.concatenate([body_feats, body_center_feats], axis=1)

        voice_feats = self.voice_feats[selected_track[self.num_face_track+self.num_body_track:]].astype(np.float32)
        voice_center_feats = []
        for tid in selected_track[self.num_face_track+self.num_body_track:]:
            fa = np.mean(self.voice_feats[self.voice_topk_track[tid][1:17]], axis=0)
            fa = fa/np.linalg.norm(fa, ord=2)
            voice_center_feats.append(fa)
        voice_center_feats = np.array(voice_center_feats, dtype=np.float32)
        voice_feats = np.concatenate([voice_feats, voice_center_feats], axis=1)
        voice_labels = self.voice_labels[selected_track[self.num_face_track+self.num_body_track:]]
        labels = np.concatenate([face_labels, body_labels, voice_labels]).astype(np.int)
        all_edges_labels = (labels.reshape(1, -1)==labels.reshape(-1, 1)).astype(np.float32)
        ignore_mask = np.zeros_like(all_edges_labels)
        if np.sum(association_matrix[len(face_feats)])<=0:
            ignore_mask[len(face_feats):len(face_feats)+self.num_per_body_track] = 1
        else:
            ignore_mask[:self.num_per_face_track] = 1
        return face_feats, body_feats, voice_feats, labels, modal_id, association_matrix, distribution_node, all_edges_labels, ignore_mask

    def __len__(self):
        if self.is_training:
            return len(self.graph_index)
        return len(self.sub_graph)
