# Introduction
Official Pytorch implementation for Semantically Structured Image Compression via Irregular Group-Based Decoupling, ICCV2023

Ruoyu Feng*, Yixin Gao*, Xin Jin, Runsen Feng, Zhibo Chen

<a href='https://arxiv.org/abs/2305.02586'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<img src=figs/motivation.png width=60% />

# Environment
```bash
conda create --name git_ssic python==3.9
conda activate git_ssic
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html  # seems to cause bug when training group-swin
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
cd data_compression
pip install -e .
cd ..
```

# Pretrained models
Download our pretrained models from [Google Drive](https://drive.google.com/drive/folders/1hmZYSdVnjyPzAx5OYaoudOpSkz3EylYG?usp=sharing) 

# Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model ours_groupswin_channelar --hyper-channels 192 192 192 \
    --lmbda 8192 --lr 5e-5 \
    --train-set /home/t2vg-a100-G4-10/project/qyp/datasets/COCO/train2017 \
    --eval-set /home/t2vg-a100-G4-10/mnt/guangtingsc_fengry/dataset/CompressionData/kodak \
    --groupvit-load-group-msk figs/kodak/group_msk \
    --total-iteration 2000000 --multistep-milestones 1600000 --eval-interval 10
```

# Inference
## For with semantic structured bitstream
```bash
  export CUDA_VISIBLE_DEVICES="0"
  python -u compress_multiple_images.py \
  compress \
  --model ours_groupswin_channelar \
  --resume path_to_checkpoint \
  --input_file_glob path_to_imgs \
  --groupvit_load_group_mask path_to_group_mask (see /figs/kodak/group_msk for example) \
  --output_file_dir codestream/bitstream \
  --verbose --hyper-channels 192 192 192 \
  --groups_tobe_decode 0 1 2 3 
```

## For without semantic structured bitstream
```bash
  export CUDA_VISIBLE_DEVICES="0"
  python -u compress_multiple_images.py \
  compress \
  --model ours_groupswin_channelar_woStructure \
  --resume path_to_checkpoint \
  --input_file_glob path_to_imgs \
  --output_file_dir logs \
  --verbose --hyper-channels 192 192 192
```

# R-D Curve of GIT-SSIC
<img src=figs/Kodak_recon_full.png width=60% />

## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{feng2023semantically,
  title={Semantically structured image compression via irregular group-based decoupling},
  author={Feng, Ruoyu and Gao, Yixin and Jin, Xin and Feng, Runsen and Chen, Zhibo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17237--17247},
  year={2023}
}
```
