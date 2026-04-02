import matplotlib.pyplot as plt
import pandas as pd
import openslide
from openslide.deepzoom import DeepZoomGenerator
import torch
import os
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
import cv2
import json
import concurrent.futures
import sys
sys.path.append('./')

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
import os
from PIL import Image
from pathlib import Path
tokenizer = get_tokenizer()


prompts = [
           'an H&E image of Low-Grade Urothelial Carcinoma', 
           'an H&E image of High-Grade Urothelial Carcinoma',
           'an H&E image of Urothelial carcinoma with histologic variants',
           'an H&E image of tumor invasion into lymphovascular',
           'an H&E image of tumor invasion into lamina propria',
           'an H&E image of tumor invasion into muscularis propria',
           'an H&E image of tumor with papillary architecture',
           'an H&E image of tumor invasion into adipose tissue']



model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="./conch/pytorch_model.bin")
_ = model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)


def process_coords_chunk(coords_chunk, tile, level_idx, scale_factor, preprocess_fn, model, text_prompts, logit_scale, device):
  
    chunk_results = []
    with torch.inference_mode():
        for y, x in coords_chunk:
            try:
                scaled_x = x // scale_factor
                scaled_y = y // scale_factor
                img = tile.get_tile(level_idx, (scaled_x, scaled_y))
                image_tensor = preprocess_fn(img).unsqueeze(0).to(device)
                image_embeddings = model.encode_image(image_tensor)
                text_embeddings = model.encode_text(text_prompts)
                sim_scores = (
                    image_embeddings @ text_embeddings.T * logit_scale.exp()
                ).softmax(dim=-1).cpu().numpy()[0]
                chunk_results.append((y, x, sim_scores))
            except Exception as e:
                if "Invalid address" in str(e):
                    continue
                else:
                    raise e
    return chunk_results


def save_normalized_patches(npy_file, he_file, output_dir):
    npy_name = os.path.splitext(os.path.basename(npy_file))[0]
    save_path = os.path.join(output_dir, f'{npy_name}.npy')
    if os.path.exists(save_path):
        print(f"Skipping {npy_file} as it already exists.")
        return

    try:
        print(f"Processing {npy_file}")
        results_map = np.load(npy_file)
        class_indices = np.argmax(results_map, axis=2)
        zero_indices = np.where(class_indices == 0)
        y_indices = zero_indices[0]
        x_indices = zero_indices[1]
        zero_probs = results_map[y_indices, x_indices, 0]
        sorted_idx = np.argsort(-zero_probs)
        y_indices = y_indices[sorted_idx]
        x_indices = x_indices[sorted_idx]
        final_coords = np.column_stack((y_indices, x_indices)).tolist()

        slide = openslide.OpenSlide(he_file)
        tile = DeepZoomGenerator(slide, tile_size=128, overlap=160, limit_bounds=False)
        score_array = np.zeros((results_map.shape[0], results_map.shape[1], len(prompts)), dtype=np.float32)


        level_idx = tile.level_count - 3
        scale_factor = 2  


        chunk_size = 1000
        coord_chunks = [final_coords[i:i+chunk_size] for i in range(0, len(final_coords), chunk_size)]
        chunk_results = []
        with concurrent.futures.ThreadPoolExecutor() as coord_executor:
            futures = []
            for coords_chunk in coord_chunks:
                futures.append(coord_executor.submit(
                    process_coords_chunk,
                    coords_chunk,
                    tile,
                    level_idx,
                    scale_factor,
                    preprocess,
                    model,
                    tokenized_prompts,
                    model.logit_scale,
                    device
                ))
            for f in concurrent.futures.as_completed(futures):
                chunk_results.extend(f.result())

        for y, x, sim_scores in chunk_results:
            score_array[y, x, :] = sim_scores

        np.save(save_path, score_array)
        print(f"Saved scores array shape: {score_array.shape} to {save_path}")

    except Exception as e:
        print(f"An error occurred while processing {npy_file}: {e}")


def find_corresponding_he_file(npy_file, he_dir):

    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    for file in os.listdir(he_dir):
        if base_name in file:
            return os.path.join(he_dir, file)
    return None
if __name__ == "__main__":
  
    npy_dir = 'path/to/UCSparseNet_npy/'
    he_dir = 'path/to/svs/'
    output_dir = 'path/to/global_save/'
    os.makedirs(output_dir, exist_ok=True)

    
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
    for npy_file in npy_files:
        npy_file_path = os.path.join(npy_dir, npy_file)
        he_file_path = find_corresponding_he_file(npy_file_path, he_dir)
        if not he_file_path:
            continue
        save_normalized_patches(npy_file_path, he_file_path, output_dir)
