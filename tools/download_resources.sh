workding_dir = "$1"

cd $workding_dir

mkdir resnet
cd resnet
wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
wget https://raw.githubusercontent.com/open-mmlab/mmpretrain/refs/heads/main/configs/_base_/models/resnet18.py
cd ..

mkdir dwpose-l
cd dwpose-l
wget https://raw.githubusercontent.com/IDEA-Research/DWPose/refs/heads/onnx/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py
gdown --id 1PHKN3p873dgCSh_YRsYqTZVj-kIbclRS
