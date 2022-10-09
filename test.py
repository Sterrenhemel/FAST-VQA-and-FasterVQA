
import argparse
from fastvqa.predict import get_score

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
