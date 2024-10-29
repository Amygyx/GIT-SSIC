# Train GIT-SSIC model.
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model ours_groupswin_channelar --hyper-channels 192 192 192 \
    --lmbda 8192 --lr 5e-5 \
    --train-set /home/t2vg-a100-G4-10/project/qyp/datasets/COCO/train2017 \
    --eval-set /home/t2vg-a100-G4-10/mnt/guangtingsc_fengry/dataset/CompressionData/kodak \
    --groupvit-load-group-msk figs/kodak/group_msk \
    --total-iteration 2000000 --multistep-milestones 1600000 --eval-interval 10