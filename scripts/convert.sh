set -ex

python ../mmdetection/tools/deployment/mmdet2torchserve.py \
                work_dir/ddod_config/ddod_config.py \
                work_dir/ddod_config/latest.pth \
                --output-folder model_store \
                --model-name Trash