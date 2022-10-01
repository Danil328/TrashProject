set -ex

python predict.py data/Frames/ \
                  work_dirs/ddog_pseudolabeling_config/ddog_pseudolabeling_config.py \
                  work_dirs/ddog_pseudolabeling_config/epoch_29.pth \
                  --out-file data/Frames/predictions.json