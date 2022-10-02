set -ex

python ../mmdetection/tools/deployment/mmdet2torchserve.py work_dirs/ddog_pseudolabeling_config/ddog_pseudolabeling_config.py work_dirs/ddog_pseudolabeling_config/epoch_15.pth \
                --output-folder model_store \
                --model-name TrashDetectonBaselineModel