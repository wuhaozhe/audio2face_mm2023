import os
import argparse
import numpy as np
import torch 
import torchaudio
import model
from dataset import DatasetProperty, get_dataset, custom_collate_fn
from render import Renderer

def load_face_mean_std(args):
    '''
        face_mean has shape of N * 3
        face_std is a float number
    '''
    face_mean = os.path.join(args.root_data_dir, args.face_mean)
    face_std = os.path.join(args.root_data_dir, args.face_std)

    # loading face mean/std
    print("loading face mean/std...")
    face_mean = torch.from_numpy(np.load(face_mean)).float()
    face_std = torch.from_numpy(np.load(face_std)).float()[0]
    if args.dataset == 'meshtalk':
        face_mean = face_mean[0]
        face_std = face_std[0]

    return face_mean, face_std

def load_mask(args):
    # load mask
    mouth_weight_fname = os.path.join(args.root_data_dir, args.mouth_weight_fname)
    upper_weight_fname = os.path.join(args.root_data_dir, args.upper_weight_fname)
    eyelid_weight_fname = os.path.join(args.root_data_dir, args.eyelid_weight_fname)
    #hair_weight_fname = os.path.join(args.root_data_dir, args.hair_weight_fname)
    # loading weight mask
    print("loading weight mask...")
    weight_mask = {}
    weight_mask['mouth']  = torch.from_numpy(np.load(mouth_weight_fname)).float().cuda().view(1, 1, -1)
    weight_mask['upper']  = torch.from_numpy(np.load(upper_weight_fname)).float().cuda().view(1, 1, -1)
    weight_mask['eyelid'] = torch.from_numpy(np.load(eyelid_weight_fname)).float().cuda().view(1, 1, -1)
    assert weight_mask['mouth'].requires_grad == False
    assert weight_mask['upper'].requires_grad == False
    assert weight_mask['eyelid'].requires_grad == False
    return weight_mask

def main(args):
    save_path = './saved_model/{}'.format(args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = './log/{}'.format(args.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    out_path = './output/{}'.format(args.name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    audio_bundle = torchaudio.pipelines.HUBERT_LARGE

    audio_fname = os.path.join(args.root_data_dir, args.audio_fname)
    template_fname = os.path.join(args.root_data_dir, args.template_fname)
    mesh_sequence_fname = os.path.join(args.root_data_dir, args.mesh_sequence_fname)

    weight_mask = load_mask(args)
    face_mean, face_std = load_face_mean_std(args)

    # prepare data
    print("preparing data...")
    dataset = get_dataset(args.dataset)

    validation_set = dataset(
        audio_fname=audio_fname,   
        template_fname=template_fname,
        meshes_fname=mesh_sequence_fname,
        audio_rate = audio_bundle.sample_rate,
        mode = 'testing',
        mean = face_mean,
        std = face_std
    )

    face_mean = face_mean.cuda()
    face_std = face_std.cuda()

    audio2face = model.Audio2FaceModel(audio_bundle, validation_set.n_vertices * 3).cuda()
    render = Renderer(validation_set.faces, validation_set.n_vertices)
    if args.dataset == 'voca':
        audio2face.load_state_dict(torch.load("saved_model/voca/model.pkl"))
    elif args.dataset == 'meshtalk':
        audio2face.load_state_dict(torch.load("saved_model/meshtalk/model.pkl"))
    elif args.dataset == 'biwi':
        audio2face.load_state_dict(torch.load("saved_model/biwi/model.pkl"))

    audio2face.eval()
    with torch.no_grad():
        for i in range(len(validation_set)):
            audio, template_tensor, mesh_tensor, _ = validation_set.__getitem__(i)
            template_tensor = template_tensor.unsqueeze(0).cuda()
            audio = audio.unsqueeze(0).cuda()
            mesh_tensor = mesh_tensor.unsqueeze(0).cuda()
            T = mesh_tensor.shape[1]

            pred_geom_self = audio2face(audio, template_tensor, mesh_tensor, T)
            pred_geom_self = (pred_geom_self.squeeze() * face_std + face_mean.unsqueeze(0).unsqueeze(1)) * 0.001
            if args.dataset == 'voca':
                pred_geom_self *= 1000
            audio_path = validation_set.get_audio_path(i)
            subj, seq = validation_set.get_subj_seq(i)
            save_path = os.path.join(out_path, "{}_{}.npy".format(subj, seq))
            np.save(save_path, pred_geom_self.detach().cpu().numpy())

            video_self_out_path = os.path.join(out_path, "{}_{}_self.mp4".format(subj, seq))
            render.to_video(pred_geom_self, audio_path, video_self_out_path, DatasetProperty.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test', help="the name of training instance")

    # training data
    parser.add_argument("--root_data_dir", type=str, default="../training_data", help="root path of data set")
    parser.add_argument("--audio_fname", type=str, default="meshes", help="subdirectory containing .wav file for audio")
    parser.add_argument("--template_fname", type=str, default="templates", help="subdirectory containing .obj file for neutral template mesh")
    parser.add_argument("--mesh_sequence_fname", type=str, default="tracked_mesh", help="subdirectory containing .npy file for mesh sequences")

    parser.add_argument("--face_mean", type=str, default="face_axis_mean.npy", help="numpy file containing face mean")
    parser.add_argument("--face_std", type=str, default="face_axis_std.npy", help="numpy file containing face std")

    parser.add_argument("--mouth_weight_fname", type=str, default="mouth_weight_mask.npy", help="weight mask for mouth area")
    parser.add_argument("--upper_weight_fname", type=str, default="upper_weight_mask.npy", help="weight mask for upper area")
    parser.add_argument("--eyelid_weight_fname", type=str, default="eyelid_weight_mask.npy", help="weight mask for eyelid area")

    # training option
    parser.add_argument("--batch_size", type=int, default=8, help="the latent feature length of audio encoder")
    parser.add_argument("--epoch", type=int, default=400, help="training epoch after each refresh")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1_norm", type=float, default=0, help="sparsity regularization")

    # dataset preprocess
    parser.add_argument("--dataset", type=str, help="name of dataset (voca, meshtalk, or biwi)")

    args = parser.parse_args()
    main(args)