INFERENCE_VIDEO="/your/path/to/inference_output.mp4"
MULTIVIEW_VIDEO_FOLDER="/your/path/to/multiview_folder"
NUM_FRAMES=25

# Step 1: Split inference video into viewXX/frame_XXXXX.png structure
python prepare_data.py --video "$INFERENCE_VIDEO" --output_dir "$MULTIVIEW_VIDEO_FOLDER" --num_frames $NUM_FRAMES

# Step 2: Copy camera templates (transforms JSON + points3d.ply)
cp -r ./cam_template/* "$MULTIVIEW_VIDEO_FOLDER"

echo "Data prepared at $MULTIVIEW_VIDEO_FOLDER"
