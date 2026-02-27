import os
from pathlib import Path
from safetensors.torch import load_file
import torch
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count

cache_dir = "cache_multiview"
root_path = Path(f"data/i2v/CharacterShot")
root_dir = root_path / cache_dir

video_latent_dir = root_dir / "video_latent"
pose_vae_dir = root_dir / "pose_latent"
pose_raw_dir = root_dir / "raw_pose"
image_dir = root_dir / "raw_image"

output_file_1 = root_path / "encoded_videos_multiview.txt"
output_file_2 = root_path / "poses_vae_multiview.txt"
output_file_3 = root_path / "poses_raw_multiview.txt"
output_file_4 = root_path / "image_raw_multiview.txt"

video_paths = sorted([file for file in video_latent_dir.glob("*.safetensors")])
pose_vae_paths = [Path(str(file).replace("video_latent", "pose_latent")) for file in video_paths]
pose_raw_paths = [Path(str(file).replace("video_latent", "raw_pose").replace(".safetensors", ".txt")) for file in video_paths]
image_paths = [Path(str(file).replace("video_latent", "raw_image").replace(".safetensors", ".txt")) for file in video_paths]

file_tuples = list(zip(video_paths, pose_vae_paths, pose_raw_paths, image_paths))

def validate_file_paths(file_tuple):
    try:
        file_path, pose_vae_path, pose_raw_path, image_path = file_tuple

        if file_path.stat().st_size == 0:
            raise RuntimeError(f"{file_path} is empty.")
        if pose_vae_path.stat().st_size == 0:
            raise RuntimeError(f"{pose_vae_path} is empty.")
        if pose_raw_path.stat().st_size == 0:
            raise RuntimeError(f"{pose_raw_path} is empty.")
        if image_path.stat().st_size == 0:
            raise RuntimeError(f"{image_path} is empty.")

        video_latent = load_file(file_path)
        pose_latent = load_file(pose_vae_path)

        pose_raw = open(pose_raw_path, 'r').read().splitlines()
        pose_raw = [line.split('.')[0] for line in pose_raw]
        
        image_raw = open(image_path, 'r').read().splitlines()
        image_raw = [line.split('.')[0] for line in image_raw]
        assert video_latent['video_latent'].shape[2] == pose_latent['pose_latent'].shape[1]

        return (
            str(file_path.relative_to(root_path)),
            str(pose_vae_path.relative_to(root_path)),
            str(pose_raw_path.relative_to(root_path)),
            str(image_path.relative_to(root_path)),
        )
    except Exception as e:
        print(f"Skipped {file_path.name} due to: {e}")
        return None

if __name__ == "__main__":
    num_workers = 8
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(validate_file_paths, file_tuples), total=len(file_tuples), desc="Validating"))

    valid_results = [res for res in results if res is not None]

    with open(output_file_1, "w") as f1, open(output_file_2, "w") as f2, open(output_file_3, "w") as f3, open(output_file_4, "w") as f4:
        for r1, r2, r3, r4 in valid_results:
            f1.write(r1 + "\n")
            f3.write(r2 + "\n")
            f4.write(r3 + "\n")
            f4.write(r4 + "\n")

    print(f"Checked {len(file_tuples)} files, {len(valid_results)} passed and saved.")
