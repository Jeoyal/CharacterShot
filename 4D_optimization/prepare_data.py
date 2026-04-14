"""
Convert inference output mp4 to the folder structure expected by 4D optimization.

Inference output: a single mp4 with 5 batches x (5 views x num_frames) frames.
View order per batch:
  batch0: [0, 1, 6, 11, 16]
  batch1: [0, 2, 7, 12, 17]
  batch2: [0, 3, 8, 13, 18]
  batch3: [0, 4, 9, 14, 19]
  batch4: [0, 5, 10, 15, 20]

Expected output structure:
  <output_dir>/
    view00/frame_00001.png ... frame_XXXXX.png
    view01/frame_00001.png ...
    ...
    view20/frame_00001.png ...
"""

import argparse
import os
import cv2
import numpy as np


VIEW_IDXS = [
    [0, 1, 6, 11, 16],
    [0, 2, 7, 12, 17],
    [0, 3, 8, 13, 18],
    [0, 4, 9, 14, 19],
    [0, 5, 10, 15, 20],
]


def split_video(video_path, output_dir, num_frames=25):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_views_per_batch = len(VIEW_IDXS[0])
    num_batches = len(VIEW_IDXS)
    expected = num_batches * num_views_per_batch * num_frames
    assert total_frames == expected, (
        f"Expected {expected} frames ({num_batches} batches x "
        f"{num_views_per_batch} views x {num_frames} frames), got {total_frames}"
    )

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Map frames to views. view0 appears in every batch; use the first occurrence.
    view_frames = {}
    idx = 0
    for batch in VIEW_IDXS:
        for view_id in batch:
            view_key = f"view{view_id:02d}"
            batch_frames = frames[idx : idx + num_frames]
            idx += num_frames
            if view_key not in view_frames:
                view_frames[view_key] = batch_frames

    # Write to disk
    for view_key in sorted(view_frames.keys()):
        view_dir = os.path.join(output_dir, view_key)
        os.makedirs(view_dir, exist_ok=True)
        for i, frame in enumerate(view_frames[view_key]):
            fname = f"frame_{i + 1:05d}.png"
            cv2.imwrite(os.path.join(view_dir, fname), frame)
        print(f"  {view_key}: {len(view_frames[view_key])} frames")

    print(f"Done. {len(view_frames)} views written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split inference mp4 into viewXX/frame_XXXXX.png structure")
    parser.add_argument("--video", type=str, required=True, help="Path to inference output mp4")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for 4D optimization")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames per view")
    args = parser.parse_args()

    split_video(args.video, args.output_dir, args.num_frames)
