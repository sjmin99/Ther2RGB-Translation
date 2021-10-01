### Using labels only
DATA_PATH=../datasets/Luckydata
# DATA_PATH=datasets/MTN_fusion

CHECKPOINTS_PATH=Unet_lucky_T2RGB_v2
# CHECKPOINTS_PATH=Hourglass_scene1_Tab2L_v2

python train.py --gpu_ids  4\
                --data_scenes Campus \
                --lr 0.0005 \
                --photometric \
                --name $CHECKPOINTS_PATH \
                --batch_size 4 \
                --load_size 512 \
                --crop_size 256 \
                --label_nc 0 \
                --input_nc 3 \
                --output_nc 3 \
                --dataroot $DATA_PATH \
                --resize_or_crop resize \
                --nThreads 4 \
                --no_instance \
                --display_freq 100 \
                --niter 100  \
                --niter_decay 50 \
                --lambda_feat 10 \
                --netG Unet\
                --use_colorjitter_contrast \
                --debug \
                # --continue_train \
                # --which_epoch latest \
                # --debug \
