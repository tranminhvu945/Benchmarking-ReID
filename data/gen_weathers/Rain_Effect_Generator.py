#!/usr/bin/env python
import os
import random
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from skimage import color
from tqdm.auto import tqdm

from lib.lime import LIME
from lib.fog_gen import fogAttenuation
from lib.rain_gen import RainGenUsingNoise
from lib.gen_utils import (illumination2opacity, 
                           layer_blend, 
                           alpha_blend, 
                           reduce_lightHSV, 
                           scale_depth)

def get_image_files(directory):
    """Get all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in Path(directory).iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--depth_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--save_folder", type=str, default="./generated/", help="path to the folder")
    parser.add_argument("--txt_file", default=None, help="path to the folder")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use, e.g., '0' or '0,1'")
    return parser.parse_args()

def set_device(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

class RainEffectGenerator:
    def __init__(self, device):
        self.device = device
        self._lime = LIME(iterations=25, alpha=1.0)
        self._illumination2darkness = {0: 1, 1: 0.95, 2: 0.85, 3: 0.8}
        self._weather2visibility = (1000, 2000)
        self._illumination2fogcolor = {0: (150, 180), 1: (180, 200), 2: (200, 240), 3: (200, 240)}
        self._rain_layer_gen = RainGenUsingNoise()
        
    def process_image(self, img_path, depth_path, save_path):
        """Process a single image and save the result"""
        try:
            rainy = self.genEffect(img_path, depth_path)
            if rainy is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rainy).save(save_path)
                print(f"Saved rain effect to: {save_path}")
                return True
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return False

    def process_directory(self, image_dir, depth_dir, save_dir):
        """Process all images in a directory"""
        image_files = get_image_files(image_dir)
        if not image_files:
            print(f"No valid images found in {image_dir}")
            return 0, 0

        success_count = 0
        fail_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {image_dir}"):
            try:
                depth_path = Path(depth_dir) / f"{img_path.stem}.npy"
                if not depth_path.exists():
                    print(f"Depth file not found: {depth_path}")
                    fail_count += 1
                    continue

                save_path = Path(save_dir) / f"{img_path.stem}.jpg"
                
                if self.process_image(img_path, depth_path, save_path):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                fail_count += 1
                
        return success_count, fail_count
    
    def getIlluminationMap(self, img: torch.Tensor) -> torch.Tensor: 
        self._lime.load(img)
        T = self._lime.illumMap()
        return torch.tensor(T, device=self.device)
    
    def getIlluminationMapCheat(self, img: torch.Tensor) -> torch.Tensor: 
        T = color.rgb2gray(img.cpu().numpy())
        return torch.tensor(T, device=self.device)
          
    def genRainLayer(self, h=256, w=256):
        blur_angle = random.choice([-1, 1])*random.randint(60, 90)
        layer_large = self._rain_layer_gen.genRainLayer(h=256, 
                                                        w=256, 
                                                        noise_scale=random.uniform(0.35, 0.55), 
                                                        noise_amount=0.2, 
                                                        zoom_layer=random.uniform(1.0, 3.5),
                                                        blur_kernel_size=random.choice([15, 17, 19, 21, 23]), 
                                                        blur_angle=blur_angle)
        
        layer_small = self._rain_layer_gen.genRainLayer(h=256, 
                                                        w=256, 
                                                        noise_scale=random.uniform(0.35, 0.55), 
                                                        noise_amount=0.15, 
                                                        zoom_layer=random.uniform(1.0, 3.5),
                                                        blur_kernel_size=random.choice([7, 9, 11, 13]), 
                                                        blur_angle=blur_angle
                                                        )
        layer = layer_blend(layer_small, layer_large)
        hl, wl = layer.shape

        if h!=hl or w!=wl:
            layer = np.asarray(Image.fromarray(layer).resize((w, h)))
        return torch.tensor(layer, device=self.device)
    
    def genEffect(self, img_path: str, depth_path: str):
        try:
            print(f"Processing image: {img_path}")
            print(f"Using depth file: {depth_path}")
            
            # CPU Operations - Image Loading and Initial Processing
            I = np.array(Image.open(img_path))
            D = np.load(depth_path)
            
            hI, wI, _ = I.shape
            hD, wD = D.shape
            
            if hI!=hD or wI!=wD:
                D = scale_depth(D, hI, wI)
            
            # GPU Operations - Heavy Computations
            I_gpu = torch.tensor(I, device=self.device).float()
            D_gpu = torch.tensor(D, device=self.device)
            
            # Illumination calculation (GPU)
            T = self.getIlluminationMapCheat(I_gpu / 255.0)
            
            # Move to CPU for histogram calculation
            T_np = T.cpu().numpy()
            illumination_array = np.histogram(T_np, bins=4, range=(0,1))[0]
            illumination_array = illumination_array / T_np.size
            illumination = illumination_array.argmax()
            
            if illumination > 0:
                visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
                fog_color = random.randint(self._illumination2fogcolor[illumination][0], 
                                         self._illumination2fogcolor[illumination][1])
                
                 # CPU operations for color transformations
                I_dark_cpu = reduce_lightHSV(I, 
                                           sat_red=self._illumination2darkness[illumination],
                                           val_red=self._illumination2darkness[illumination])
                
                # Fog effect on CPU
                I_fog = fogAttenuation(I_dark_cpu, D, visibility=visibility, fog_color=fog_color)
                # Convert result back to GPU
                I_fog = torch.tensor(I_fog, device=self.device)
            else:
                fog_color = 75
                D_max = D_gpu.max().item()
                visibility = D_max * 0.75 if D_max < 1000 else 750
                # Move to CPU for fog effect
                I_cpu = I_gpu.cpu().numpy()
                D_cpu = D_gpu.cpu().numpy()
                I_fog = fogAttenuation(I_cpu, D_cpu, visibility=visibility, fog_color=fog_color)
                # Back to GPU
                I_fog = torch.tensor(I_fog, device=self.device)
            
            # Generate rain effect (GPU operation)
            rain_layer = self.genRainLayer(h=hI, w=wI)
            
            # Final blending - move everything to CPU
            I_fog_cpu = I_fog.cpu().numpy()
            rain_layer_cpu = rain_layer.cpu().numpy()
            
            # CPU operations
            alpha = illumination2opacity(I, illumination) * random.uniform(0.3, 0.5)

            # Final blend on CPU
            I_rain = alpha_blend(I_fog_cpu, rain_layer_cpu, alpha)
            
            return I_rain.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in genEffect for {img_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def process_batch(self, img_paths, depth_paths, save_paths, batch_size=32):
        """Process images in batches to better utilize GPU"""
        for i in range(0, len(img_paths), batch_size):
            batch_imgs = img_paths[i:i + batch_size]
            batch_depths = depth_paths[i:i + batch_size]
            batch_saves = save_paths[i:i + batch_size]
            
            for img_path, depth_path, save_path in zip(batch_imgs, batch_depths, batch_saves):
                try:
                    rainy = self.genEffect(img_path, depth_path)
                    if rainy is not None:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rainy).save(save_path)
                        print(f"Saved: {save_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
                
            # Optional: Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main():
    args = parse_arguments()
    device = set_device(args.gpu_ids)
    raingen = RainEffectGenerator(device)
    
    clear_path = Path(args.clear_path)
    depth_path = Path(args.depth_path)
    save_folder = Path(args.save_folder)
    
    if not clear_path.exists():
        raise ValueError(f"Input path does not exist: {clear_path}")
    if not depth_path.exists():
        raise ValueError(f"Depth path does not exist: {depth_path}")
    
    save_folder.mkdir(parents=True, exist_ok=True)
    
    success_total = 0
    fail_total = 0
    
    if clear_path.is_file():
        if depth_path.is_file() and depth_path.suffix == ".npy":
            success = raingen.process_image(clear_path, depth_path, 
                                         save_folder / f"{clear_path.stem}.jpg")
            success_total = 1 if success else 0
            fail_total = 0 if success else 1
    else:
        print(f"\nProcessing directory: {clear_path}")
        success_total, fail_total = raingen.process_directory(clear_path, depth_path, save_folder)
    
    print(f"\nProcessing completed:")
    print(f"Successfully processed images: {success_total}")
    print(f"Failed images: {fail_total}")

if __name__ == '__main__':
    main()