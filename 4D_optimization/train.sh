MULTIVIEW_VIDEO_FOLDER="/your/path/to/multiview_folder"

python train.py -s "$MULTIVIEW_VIDEO_FOLDER" --port 6017 --expname "test" --configs ./arguments/dnerf/dnerf_default.py
