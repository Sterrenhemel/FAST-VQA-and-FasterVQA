
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np
import argparse
from .predict import get_score


a1, a2, a3, a4 = -0.08283314, 0.037915636, 17.057644220869555, 63.291117743589744
def rescale(x):
    return ((x - a1) / a2) * a3 + a4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ### can choose between
    ### options/fast/f3dvqa-b.yml
    ### options/fast/fast-b.yml
    ### options/fast/fast-m.yml
    parser.add_argument(
        "-o", "--opt", type=str, 
        default="./options/fast/fast-b.yml", 
        help="the option file"
    )
    
    ## can be your own
    parser.add_argument(
        "-v", "--video_path", type=str, 
        default="./demos/10053703034.mp4", 
        help="the input video path"
    )
    
    parser.add_argument(
        "-d", "--device", type=str, 
        default="cuda", 
        help="the running device"
    )
    
    
    args = parser.parse_args()

    score = get_score(args.video_path, args.device)
    print(f"The quality score of the video is {score:.5f}.")
