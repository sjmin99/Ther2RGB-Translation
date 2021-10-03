### Using labels only
DATA_PATH=../datasets/Luckydata
# DATA_PATH=datasets/MTN_fusion

CHECKPOINTS_PATH=Unet_lucky_T2ab_noise
# CHECKPOINTS_PATH=Hourglass_scene1_Tab2L_v2

python train.py --gpu_ids  1,2\
                --data_scenes Campus \
                --lr 0.0005 \
                --photometric \
                --name $CHECKPOINTS_PATH \
                --batch_size 16 \
                --load_size 256 \
                --crop_size 256 \
                --label_nc 0 \
                --input_nc 3 \
                --output_nc 2 \
                --dataroot $DATA_PATH \
                --resize_or_crop resize \
                --nThreads 16 \
                --no_instance \
                --display_freq 100 \
                --niter 150  \
                --niter_decay 50 \
                --lambda_feat 10 \
                --netG Unet\
                --no_vgg_loss --no_gan_loss --no_ganFeat_loss \
                --use_noise \
#                 --debug \
                # --continue_train \
                # --which_epoch 150 \
                # --debug \
