
from re import S
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np


a1, a2, a3, a4 = -0.08283314, 0.037915636, 17.057644220869555, 63.291117743589744
def rescale(x):
    return ((x - a1) / a2) * a3 + a4


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


@run_once
def get_model(opt, device):
    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    model = torch.utils.model_zoo.load_url(
        url="https://github.com/TimothyHTimothy/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth",
        map_location=device, 
        progress=True,
    )
    evaluator.load_state_dict(model["state_dict"])
    return evaluator


def get_score(video_path: str, device: str):
    video_reader = decord.VideoReader(video_path)
    opt = {'name': 'FAST-VQA-B-Refactor-1*4', 'num_epochs': 30, 'l_num_epochs': 0, 'warmup_epochs': 2.5, 'ema': True, 'save_model': True, 'batch_size': 16, 'num_workers': 6, 'wandb': {'project_name': 'VQA_Experiments_2022'}, 'data': {'train': {'type': 'FusionDataset', 'args': {'phase': 'train', 'anno_file': './examplar_data_labels/train_labels.txt', 'data_prefix': '../datasets/LSVQ', 'sample_types': {'fragments': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32}}, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 1}}, 'val-livevqc': {'type': 'FusionDataset', 'args': {'phase': 'test', 'anno_file': './examplar_data_labels/LIVE_VQC/labels.txt', 'data_prefix': '../datasets/LIVE_VQC/', 'sample_types': {'fragments': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32}}, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 4}}, 'val-kv1k': {'type': 'FusionDataset', 'args': {'phase': 'test', 'anno_file': './examplar_data_labels/KoNViD/labels.txt', 'data_prefix': '../datasets/KoNViD/', 'sample_types': {'fragments': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32}}, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 4}}, 'val-ltest': {'type': 'FusionDataset', 'args': {'phase': 'test', 'anno_file': './examplar_data_labels/LSVQ/labels_test.txt', 'data_prefix': '../datasets/LSVQ/', 'sample_types': {'fragments': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32}}, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 4}}, 'val-l1080p': {'type': 'FusionDataset', 'args': {'phase': 'test', 'anno_file': './examplar_data_labels/LSVQ/labels_1080p.txt', 'data_prefix': '../datasets/LSVQ/', 'sample_types': {'fragments': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32}}, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 4}}}, 'model': {'type': 'DiViDeAddEvaluator', 'args': {'backbone': {'fragments': {'checkpoint': False, 'pretrained': None}}, 'backbone_size': 'swin_tiny_grpb', 'backbone_preserve_keys': 'fragments', 'divide_head': False, 'vqa_head': {'in_channels': 768, 'hidden_channels': 64}}}, 'optimizer': {'lr': 0.001, 'backbone_lr_mult': 0.1, 'wd': 0.05}, 'load_path': '../pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth', 'test_load_path': './pretrained_weights/FAST_VQA_B_1_4.pth'}

    ### Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag",1) > 1:
            sampler = FragmentSampleFrames(fsize_t=t_data_opt["clip_len"] // t_data_opt.get("t_frag",1),
                                           fragments_t=t_data_opt.get("t_frag",1),
                                           num_clips=t_data_opt.get("num_clips",1),
                                          )
        else:
            sampler = SampleFrames(clip_len = t_data_opt["clip_len"], num_clips = t_data_opt["num_clips"])
        frames = sampler(len(video_reader))
        print("Sampled frames are", frames)
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        num_clips = t_data_opt.get("num_clips",1)
        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
        vsamples[sample_type] = sampled_video.to(args.device)
    
    evaluator = get_model()
    result = evaluator(vsamples)
    score = rescale(result.mean().item())
    return score
