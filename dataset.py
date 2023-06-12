import os
import torch
import torchaudio
import pickle
from random import randint
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.io import load_ply, load_obj

class DatasetProperty():
    fps = None
    audio_rate = None

def get_dataset(dataset_name):
    if dataset_name == 'voca':
        return VOCASet
    elif dataset_name == 'meshtalk':
        return MeshTalkSet
    elif dataset_name == 'biwi':
        return BIWIset
    else:
        raise Exception('unknown dataset')

# 在dataload期间就做了标准化
voca_div = {
    'subject_for_training':
    '''
    FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA
    FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA
    ''',
    'sequence_for_training':
    '''
            sentence01 sentence02 sentence03 sentence04 sentence05 sentence06 sentence07 sentence08 sentence09 sentence10
            sentence11 sentence12 sentence13 sentence14 sentence15 sentence16 sentence17 sentence18 sentence19 sentence20
            sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30
            sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40
    ''',
    'subject_for_validation': "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA",
    'sequence_for_validation':
    '''
            sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 
            sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40
    ''',
    'subject_for_testing': "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA ",
    'sequence_for_testing':
    '''
            sentence01 sentence02 sentence03 sentence04 sentence05 sentence06 sentence07 sentence08 sentence09 sentence10
            sentence11 sentence12 sentence13 sentence14 sentence15 sentence16 sentence17 sentence18 sentence19 sentence20
            sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30
            sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40
    '''
}

class VOCASet(Dataset):
    def __init__(self, audio_fname, meshes_fname, template_fname, audio_rate, mode='training', mean = None, std = None):
        self.mode = mode # training, validation, testing

        print(f"loading audio from {audio_fname}")
        self.raw_audio_dir = audio_fname

        print(f"loading meshes from {meshes_fname}")
        self.mesh_dir = meshes_fname
        self.mesh_sequences = {}
        self.mesh_sequences = pickle.load(open(meshes_fname, 'rb'))

        print(f"loading templates from {template_fname}")
        self.template_dir = template_fname
        self.templates = {}
        self.n_vertices = None
        self.faces = None

        self.audio_rate = audio_rate
        if mean is None:
            self.mean = torch.zeros(3).float()
        else:
            self.mean = mean

        if std is None:
            self.std = 1
        else:
            self.std = std

        DatasetProperty.fps = 60
        DatasetProperty.audio_rate = audio_rate

        self._init_indices()
        print(f"{len(self)} samples loaded")

    def _init_indices(self):
        self.indices = []
        self.subj_id_mapping = {}
        #for subj in self.raw_audio:
            #for seq in self.raw_audio[subj]:
        for i, subj in enumerate(voca_div[f"subject_for_{self.mode}"].split()):
            self.subj_id_mapping[subj] = i
            try:
                self.templates[subj], self.faces = load_ply(os.path.join(self.template_dir, subj + '.ply'))
                if self.n_vertices is None:
                    self.n_vertices = self.templates[subj].shape[0]
                #print(self.templates[subj])
            except FileNotFoundError:
                print(f"missing - {subj}") 
            for seq in voca_div[f"sequence_for_{self.mode}"].split():
                try:
                    if self.mesh_sequences[subj][seq] is not None and self.templates[subj] is not None:
                        self.indices.append((subj, seq))
                except KeyError:
                    print(f"missing - {subj} {seq}")

    def get_audio_path(self, idx):
        subj, seq = self.indices[idx]
        audio_path = os.path.join(self.raw_audio_dir, subj, seq + '.wav')
        return audio_path

    def get_subj_seq(self, idx):
        subj, seq = self.indices[idx]
        return subj, seq

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        subj, seq = self.indices[idx]

        audio_path = os.path.join(self.raw_audio_dir, subj, seq + '.wav')
        audio, sample_rate = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sample_rate, self.audio_rate)[0]

        template_tensor = (self.templates[subj] - self.mean.unsqueeze(0)) / self.std
        mesh_tensor = (self.mesh_sequences[subj][seq] - self.mean.unsqueeze(0).unsqueeze(1)) / self.std
        subj_id = self.subj_id_mapping[subj]

        return audio, template_tensor, mesh_tensor, subj_id

meshtalk_div = {
    'subject_for_training':
    '''
    m--20180227--0000--6795937--GHS m--20180927--0000--7889059--GHS m--20180406--0000--8870559--GHS m--20181017--0000--002914589--GHS m--20171024--0000--002757580--GHS m--20180418--0000--2183941--GHS m--20190529--1004--5067077--GHS m--20180105--0000--002539136--GHS m--20190828--1318--002645310--GHS
    ''',
    'subject_for_validation': "m--20180226--0000--6674443--GHS m--20180510--0000--5372021--GHS m--20180426--0000--002643814--GHS m--20190529--1300--002421669--GHS",
    'subject_for_testing': "m--20180426--0000--002643814--GHS m--20190529--1300--002421669--GHS"
}

class MeshTalkSet(Dataset):
    def __init__(self, audio_fname, meshes_fname, template_fname, audio_rate, mode='training', mean = None, std = None):
        self.mode = mode # training, validation, testing

        print(f"loading audio from {audio_fname}")
        self.raw_audio_dir = audio_fname

        print(f"loading meshes from {meshes_fname}")
        self.mesh_dir = meshes_fname
        self.mesh_sequences = {}
        for subj in os.listdir(self.mesh_dir):
            self.mesh_sequences[subj] = {}
            for seq in os.listdir(os.path.join(self.mesh_dir, subj)):
                self.mesh_sequences[subj][seq[:-4]] = torch.from_numpy(np.load(os.path.join(self.mesh_dir, subj, seq)))


        print(f"loading templates from {template_fname}")
        self.template_dir = template_fname
        self.templates = {}
        self.n_vertices = None
        self.faces = None

        self.audio_rate = audio_rate
        if mean is None:
            self.mean = torch.zeros(3).float()
        else:
            self.mean = mean

        if std is None:
            self.std = 1
        else:
            self.std = std

        DatasetProperty.fps = 30
        DatasetProperty.audio_rate = audio_rate

        self._init_indices()
        print(f"{len(self)} samples loaded")

    def _init_indices(self):
        self.indices = []
        self.subj_id_mapping = {}
        for i, subj in enumerate(meshtalk_div[f"subject_for_{self.mode}"].split()):
            self.subj_id_mapping[subj] = i
            try:
                self.templates[subj], self.faces, _ = load_obj(os.path.join(self.template_dir, subj + '.obj'))
                self.faces = self.faces[0]
                if self.n_vertices is None:
                    self.n_vertices = self.templates[subj].shape[0]
            except FileNotFoundError:
                print(f"missing - {subj}") 
            for seq in self.mesh_sequences[subj].keys():
                self.indices.append((subj, seq))
    
    def get_audio_path(self, idx):
        subj, seq = self.indices[idx]
        audio_path = os.path.join(self.raw_audio_dir, subj, seq + '.wav')
        return audio_path

    def get_subj_seq(self, idx):
        subj, seq = self.indices[idx]
        return subj, seq

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        subj, seq = self.indices[idx]

        audio_path = os.path.join(self.raw_audio_dir, subj, seq + '.wav')
        audio, sample_rate = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sample_rate, self.audio_rate)[0]

        template_tensor = (self.templates[subj] - self.mean.unsqueeze(0)) / self.std
        mesh_tensor = (self.mesh_sequences[subj][seq] - self.mean.unsqueeze(0).unsqueeze(1)) / self.std
        subj_id = self.subj_id_mapping[subj]

        return audio, template_tensor, mesh_tensor, subj_id

biwi_div = {
    'subject_for_training': "F2 F3 F4 M3 M4 M5" ,
    'subject_for_validation': "F2 F3 F4 M3 M4 M5",
    'subject_for_testing': "F1 F5 F6 F7 F8 M1 M2 M6",
    'sequence_for_training': "e01 e02 e03 e04 e05 e06 e07 e08 e09 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25 e26 e27 e28 e29 e30 e31 e32",
    'sequence_for_validation': " e33 e34 e35 e36",
    'sequence_for_testing': "e37 e38 e39 e40"
}

dirty_data = '''
    F1_36
    F1_39
    F1_e20 
    F3_e02
    F3_e03
    F4_27
    F6_36
    M2_20
    M6_21
    F1_e12
    F6_e21
    F6_e31
    M2_31
    M2_e04
    M2_e08
    M6_e02
    M6_e18
    M6_e27
    M1_e01
    M1_e02
    M1_e03
    M1_e04
    M1_e05
    M1_e06
    M1_e07
    M1_e08
    M1_e09
    M1_e10
    M1_e11
    M1_e12
    M1_e13
    M1_e14
    M1_e15
    M1_e16
    M1_e17
    M1_e18
    M1_e19
    M1_e20
    '''

class BIWIset(Dataset):
    def __init__(self, audio_fname, meshes_fname, template_fname, audio_rate, mode='training', mean = None, std = None):
        self.mode = mode # training, validation, testing

        print(f"loading audio from {audio_fname}")
        self.raw_audio_dir = audio_fname

        self.dirty_data = dirty_data.split()
        print(f"loading meshes from {meshes_fname}")
        self.mesh_dir = meshes_fname
        self.mesh_sequences = {}
        # dirty_data_count = 0
        # valid_data_count = 0
        # for subj in os.listdir(self.mesh_dir):
        #     self.mesh_sequences[subj] = {}
        #     print(subj)
        #     for seq in os.listdir(os.path.join(self.mesh_dir, subj)):
        #         if not f"{subj}_{seq[:-4]}" in self.dirty_data and seq.startswith('e'):
        #             self.mesh_sequences[subj][seq[:-4]] = torch.from_numpy(np.load(os.path.join(self.mesh_dir, subj, seq)))
        #             valid_data_count += 1
        #         else:
        #             dirty_data_count += 1

        # print(f"{valid_data_count} valid samples found, {dirty_data_count} samples disposed.")

        print(f"loading templates from {template_fname}")
        self.template_dir = template_fname
        self.templates = {}
        self.n_vertices = None
        self.faces = None

        self.audio_rate = audio_rate
        if mean is None:
            self.mean = torch.zeros(3).float()
        else:
            self.mean = mean

        if std is None:
            self.std = 1
        else:
            self.std = std

        DatasetProperty.fps = 25
        DatasetProperty.audio_rate = audio_rate

        self._init_indices()
        print(f"{len(self)} samples loaded")

    def _init_indices(self):
        self.indices = []
        self.subj_id_mapping = {}
        for i, subj in enumerate(biwi_div[f"subject_for_{self.mode}"].split()):
            self.subj_id_mapping[subj] = i
            try:
                self.templates[subj], self.faces = load_ply(os.path.join(self.template_dir, subj + '.ply'))
                if self.n_vertices is None:
                    self.n_vertices = self.templates[subj].shape[0]
                #print(self.templates[subj])
            except FileNotFoundError:
                print(f"missing - {subj}")
            self.mesh_sequences[subj] = {}
            for seq in biwi_div[f"sequence_for_{self.mode}"].split():
                try:
                    if self.templates[subj] is not None:
                        self.mesh_sequences[subj][seq] = torch.from_numpy(np.load(os.path.join(self.mesh_dir, subj, seq + '.npy'))).float()
                        self.indices.append((subj, seq))
                except FileNotFoundError:
                    print(f"missing - {subj} {seq}")

    def get_audio_path(self, idx):
        subj, seq = self.indices[idx]
        audio_path = os.path.join(self.raw_audio_dir, subj + '_' + seq + '_cut.wav')
        return audio_path
    
    def get_subj_seq(self, idx):
        subj, seq = self.indices[idx]
        return subj, seq

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        subj, seq = self.indices[idx]

        audio_path = os.path.join(self.raw_audio_dir, subj + '_' + seq + '_cut.wav')
        audio, sample_rate = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sample_rate, self.audio_rate)[0]

        template_tensor = (self.templates[subj] - self.mean.unsqueeze(0)) / self.std
        mesh_tensor = (self.mesh_sequences[subj][seq] - self.mean.unsqueeze(0).unsqueeze(1)) / self.std
        subj_id = self.subj_id_mapping[subj]

        return audio, template_tensor, mesh_tensor, subj_id

def custom_collate_fn(batch_data):
    min_mesh_len = None
    min_audio_len = None
    for audio, _, mesh, _ in batch_data:
        min_mesh_len = len(mesh) if min_mesh_len is None else min(min_mesh_len, len(mesh))
        min_audio_len = len(audio) if min_audio_len is None else min(min_audio_len, len(audio))

    min_mesh_len = min(min_mesh_len, int(min_audio_len * DatasetProperty.fps / DatasetProperty.audio_rate))
    min_audio_len = int(min_mesh_len * DatasetProperty.audio_rate / DatasetProperty.fps)

    mesh_tensor = []
    audio_tensor = []
    mesh_id_tensor = []
    mesh_template_tensor = []

    for audio, template, mesh, subj_id in batch_data:
        rand_bound = min(len(mesh), int(len(audio) * DatasetProperty.fps / DatasetProperty.audio_rate))
        mesh_start_index = randint(0, rand_bound - min_mesh_len)
        mesh_end_index = mesh_start_index + min_mesh_len

        audio_start_idx = int(mesh_start_index * DatasetProperty.audio_rate / DatasetProperty.fps)
        audio_end_idx = audio_start_idx + min_audio_len

        mesh_tensor.append(mesh[mesh_start_index: mesh_end_index])
        audio_crop = audio[audio_start_idx: audio_end_idx]
        audio_tensor.append(audio_crop)
        mesh_id_tensor.append(subj_id)
        mesh_template_tensor.append(template)

    mesh_tensor = torch.stack(mesh_tensor, dim = 0)
    audio_tensor = torch.stack(audio_tensor, dim = 0)
    mesh_id_tensor = torch.LongTensor(mesh_id_tensor)
    mesh_template_tensor = torch.stack(mesh_template_tensor, dim = 0)

    return audio_tensor, mesh_template_tensor, mesh_tensor, mesh_id_tensor