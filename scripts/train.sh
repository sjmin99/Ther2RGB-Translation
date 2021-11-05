DATA_PATH="./T2R_Dataset"
MODEL_NAME=THER2RGB

python train.py --gpu_ids  4\
                --lr 0.0005 \
                --photometric \
                --name $MODEL_NAME \
                --batch_size 4 \
                --load_size 512 \
                --crop_size 256 \
                --dataroot $DATA_PATH \
                --resize_or_crop resize \
                --nThreads 4 \
                --display_freq 100 \
                --niter 100  \
                --niter_decay 50 \
                --lambda_feat 10 \
