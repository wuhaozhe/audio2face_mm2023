import os
import argparse
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import model
import utils
from random import randint
from dataset import DatasetProperty, get_dataset, custom_collate_fn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from render import Renderer
from tqdm import tqdm

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.val * 0.9 + val * 0.1

    def __call__(self):
        return self.val
    
def load_face_mean_std(args):
    '''
        the returned face_mean is a vector with length 3, which respectively denotes the mean of x, y, z
        the returned face_std is a float number, which denotes the overall std 
    '''
    face_mean = os.path.join(args.root_data_dir, args.face_mean)
    face_std = os.path.join(args.root_data_dir, args.face_std)

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

    save_path = './saved_model/{}'.format(args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = './log/{}'.format(args.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    out_path = './output/{}'.format(args.name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    audio_fname = os.path.join(args.root_data_dir, args.audio_fname)
    template_fname = os.path.join(args.root_data_dir, args.template_fname)
    mesh_sequence_fname = os.path.join(args.root_data_dir, args.mesh_sequence_fname)

    weight_mask = load_mask(args)
    face_mean, face_std = load_face_mean_std(args)

    # prepare data
    print("preparing data...")
    dataset = get_dataset(args.dataset)

    training_set = dataset(
        audio_fname=audio_fname,   
        template_fname=template_fname,
        meshes_fname=mesh_sequence_fname,
        audio_rate = audio_bundle.sample_rate,
        mode = 'training',
        mean = face_mean,
        std = face_std
    )   
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

    training_dataloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=False, drop_last=True, collate_fn=custom_collate_fn)

    # prepare model and optimizer
    audio2face = model.Audio2FaceModel(audio_bundle, training_set.n_vertices * 3).cuda()
    audio2face.freeze_pretrain()
    optimizer = torch.optim.Adam([
            {'params': audio2face.parameters()}
        ], lr = args.lr)
    criterion = nn.MSELoss(reduction='none')
    mesh_error = utils.MeshLoss()
    render = Renderer(training_set.faces, training_set.n_vertices)

    # prepare logger
    writer = SummaryWriter(log_path)
    loss_logs = {
        'recon': AverageMeter(),
        'reg': AverageMeter(),
        'all': AverageMeter()
    }

    iter_idx = 0
    for epoch in range(args.epoch):
        audio2face.train()
        for audio_batch, template_batch, mesh_batch, _ in tqdm(training_dataloader):
            audio_batch, template_batch, mesh_batch = audio_batch.cuda(), template_batch.cuda(), mesh_batch.cuda()

            B, T = mesh_batch.shape[0], mesh_batch.shape[1]

            pred_geom = audio2face(audio_batch, template_batch, mesh_batch, T)

            optimizer.zero_grad()
            recon_loss = criterion(pred_geom, mesh_batch).sum(dim = -1)
            recon_loss = recon_loss.sum() / (B * T)
            if args.dataset == 'biwi':
                recon_loss = recon_loss * 0.1
            reg_loss = audio2face.regularizer() * 1e-4

            loss_all = recon_loss + reg_loss
            loss_all.backward()
            optimizer.step()

            loss_logs['recon'].update(recon_loss.item())
            loss_logs['reg'].update(reg_loss.item())
            loss_logs['all'].update(loss_all.item())

            if iter_idx % 50 == 0 and iter_idx != 0:
                print("iter {}".format(iter_idx), end=' ')
                for key, value in loss_logs.items():
                    print("{} {}".format(key, value()), end = ' ')
                    writer.add_scalar(key, value(), iter_idx)
                print()
            
            if iter_idx % 1000 == 0 and iter_idx != 0:
                audio2face.eval()
                # since we conduct normalization in the dataset, we need to conduct denormalization in the evaluation
                evaluate_error = {
                    'upper_max_self': [], 
                    'upper_mean_self': [], 
                    'mouth_max_self': [], 
                    'mouth_mean_self': [],
                    'mouth_dtw_self': [],
                    'recon_max_self': [],
                    'recon_mean_self': [],
                    'recon_dtw_self': [],
                    'upper_max_style': [], 
                    'upper_mean_style': [], 
                    'mouth_max_style': [], 
                    'mouth_mean_style': [],
                    'mouth_dtw_style': [],
                    'recon_max_style': [],
                    'recon_mean_style': [],
                    'recon_dtw_style': [],
                }

                with torch.no_grad():
                    for i in range(len(validation_set)):
                        audio, template_tensor, mesh_tensor, _ = validation_set.__getitem__(i)
                        audio = audio.unsqueeze(0).cuda()
                        template_tensor = template_tensor.unsqueeze(0).cuda()
                        mesh_tensor = mesh_tensor.unsqueeze(0).cuda()
                        T = mesh_tensor.shape[1]
                        pred_geom_self = audio2face(audio, template_tensor, mesh_tensor, T)
                        pred_geom_style = None
                        min_recon_error = None
                        for j in range(0, len(training_set), 10):
                            _, random_template_tensor, random_mesh_tensor, _ = training_set.__getitem__(randint(0, len(training_set) - 1))
                            random_mesh_tensor = random_mesh_tensor.unsqueeze(0).cuda()
                            random_template_tensor = random_template_tensor.unsqueeze(0).cuda()
                            T = mesh_tensor.shape[1]
                            rand_pred_geom = audio2face(audio, template_tensor, random_mesh_tensor - random_template_tensor.unsqueeze(1) + template_tensor.unsqueeze(1), T)
                            if pred_geom_style is None:
                                min_recon_error = mesh_error.compute_geometric_mean_euclidean_dist_error(mesh_tensor, rand_pred_geom).item()
                                pred_geom_style = rand_pred_geom
                            else:
                                tmp_error = mesh_error.compute_geometric_mean_euclidean_dist_error(mesh_tensor, rand_pred_geom).item()
                                if tmp_error < min_recon_error:
                                    min_recon_error = tmp_error
                                    pred_geom_style = rand_pred_geom
                                else:
                                    del random_mesh_tensor, random_template_tensor, rand_pred_geom
                        
                        if i < 5:
                            audio_path = validation_set.get_audio_path(i)
                            video_out_path = os.path.join(out_path, "{}_{}_{}.mp4".format(i, iter_idx, args.dataset))
                            rand_video_out_path = os.path.join(out_path, "{}_{}_{}_rand.mp4".format(i, iter_idx, args.dataset))
                            video_gt_out_path = os.path.join(out_path, "{}_{}_{}_gt.mp4".format(i, iter_idx, args.dataset))
                            if args.dataset == 'meshtalk':
                                render_scale = 0.15
                            elif args.dataset == 'voca':
                                render_scale = 0.05
                            elif args.dataset == 'biwi':
                                render_scale = 0.05
                            render.to_video(pred_geom_self.squeeze() * render_scale, audio_path, video_out_path, DatasetProperty.fps)
                            render.to_video(mesh_tensor.squeeze() * render_scale, audio_path, video_gt_out_path, DatasetProperty.fps)
                            render.to_video(pred_geom_style.squeeze() * render_scale, audio_path, rand_video_out_path, DatasetProperty.fps)

                        mesh_tensor = mesh_tensor * face_std + face_mean.view(1, 1, 1, -1)
                        for j in range(2):
                            if j == 0:
                                pred_geom = pred_geom_self * face_std + face_mean.view(1, 1, 1, -1)
                                prefix = 'self'
                            else:
                                pred_geom = pred_geom_style * face_std + face_mean.view(1, 1, 1, -1)
                                prefix = 'style'
        
                            upper_mean_error = mesh_error.compute_geometric_mean_euclidean_dist_error(mesh_tensor, pred_geom, weight_mask['upper'])
                            upper_max_error = mesh_error.compute_geometric_maximal_euclidean_dist(mesh_tensor, pred_geom, weight_mask['upper'])
                            mouth_mean_error = mesh_error.compute_geometric_mean_euclidean_dist_error(mesh_tensor, pred_geom, weight_mask['mouth'])
                            mouth_max_error = mesh_error.compute_geometric_maximal_euclidean_dist(mesh_tensor, pred_geom, weight_mask['mouth'])
                            mouth_dtw_error = mesh_error.dtw_error(mesh_tensor.squeeze().cpu(), pred_geom.squeeze().cpu(), weight_mask['mouth'].cpu())
                            recon_mean_error = mesh_error.compute_geometric_mean_euclidean_dist_error(mesh_tensor, pred_geom)
                            recon_max_error = mesh_error.compute_geometric_maximal_euclidean_dist(mesh_tensor, pred_geom)
                            recon_dtw_error = mesh_error.dtw_error(mesh_tensor.squeeze().cpu(), pred_geom.squeeze().cpu())
                            evaluate_error['upper_mean' + '_' + prefix].append(upper_mean_error.item())
                            evaluate_error['upper_max' + '_' + prefix].append(upper_max_error.item())
                            evaluate_error['mouth_mean' + '_' + prefix].append(mouth_mean_error.item())
                            evaluate_error['mouth_max' + '_' + prefix].append(mouth_max_error.item())
                            evaluate_error['mouth_dtw' + '_' + prefix].append(mouth_dtw_error)
                            evaluate_error['recon_mean' + '_' + prefix].append(recon_mean_error.item())
                            evaluate_error['recon_max' + '_' + prefix].append(recon_max_error.item())
                            evaluate_error['recon_dtw' + '_' + prefix].append(recon_dtw_error)

                        del pred_geom_style, pred_geom_self, audio, template_tensor, mesh_tensor
                        
                    print("evaluate {}".format(iter_idx), end=' ')
                    for key, value in evaluate_error.items():
                        mean_value = np.mean(value)
                        print("{} {}".format(key + 'error', mean_value), end=' ')
                        writer.add_scalar(key + 'error', mean_value, iter_idx)
                    print()

                torch.save(audio2face.state_dict(), os.path.join(save_path, 'model.pkl'))
                print("save model at iter {}".format(iter_idx))

                audio2face.train()

            iter_idx += 1

        if epoch == 10:
            print("unfreeze hubert")
            audio2face.unfreeeze_pretain()

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