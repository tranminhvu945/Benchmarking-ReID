import os
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to the file or the folder")
    parser.add_argument("--save_folder", type=str, default="./depth/", help="Path to the folder")
    parser.add_argument("--midas_model", type=str, default="DPT_Large", help="Midas model name")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use, e.g., '0' or '0,1'")
    parser.add_argument("--baseline", type=float, default=0.54, help="Baseline distance for depth calculation")
    parser.add_argument("--focal", type=float, default=721.09, help="Focal length for depth calculation")
    parser.add_argument("--img_scale", type=float, default=1, help="Image scale factor")
    return parser.parse_args()

def set_device(use_cuda, gpu_ids):
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def get_depth_estimation_model(model_name: str, device="cpu"):
    assert model_name in ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.eval()
    midas.to(device)
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform

def get_disparity_map(model, transform, img_path, device):
    img = Image.open(img_path).convert('RGB')
    
    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(img.shape[2], img.shape[3]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def main():
    args = parse_arguments()
    
    device = set_device(args.use_cuda, args.gpu_ids)
        
    baseline = args.baseline
    focal = args.focal
    img_scale = args.img_scale
    
    imgP = Path(args.img_path)
    save_folder = Path(args.save_folder)
    if not save_folder.exists():
        os.makedirs(str(save_folder))
    
    midas, midas_transform = get_depth_estimation_model(model_name=args.midas_model, device=device)
    
    if imgP.is_file():
        disp = get_disparity_map(midas, midas_transform, imgP, device)
        disp[disp < 0] = 0
        disp = disp + 1e-3
        depth = baseline * focal / (disp * img_scale)
        np.save(save_folder / imgP.stem, depth)

    if imgP.is_dir():
        print(f"Processing directory: {imgP}")
        image_files = sorted(imgP.glob("*"))
        for imgp in tqdm(image_files):
            save_path = save_folder / (imgp.stem + ".npy")
            if save_path.exists():
                print(f"Skipping existing depth file: {save_path}")
                continue
            print(f"Processing file: {imgp}")
            disp = get_disparity_map(midas, midas_transform, imgp, device)
            disp[disp < 0] = 0
            disp = disp + 1e-3
            depth = baseline * focal / (disp * img_scale)

            depth = np.clip(depth * 1.2, 0, 255)
            
            np.save(save_path, depth)
            print(f"Saved depth file: {save_path}")

if __name__ == '__main__':
    main()
