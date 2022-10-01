set -ex

CUDA_VISIBLE_DEVICES="4,5,6,7" bash ../mmdetection/tools/dist_train.sh configs/ddod_config.py 4