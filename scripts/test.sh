################################ Testing ################################
# labels only
DATA_PATH=../datasets/Luckydata
# DATA_PATH=datasets/
CHECKPOINTS_PATH=Unet_lucky_T2RGB_v1

python test.py --name $CHECKPOINTS_PATH \
               --dataroot $DATA_PATH \
               --gpu_ids 4 \
               --load_size 512 \
               --nThreads 1\
               --no_flip \
               --serial_batches \
               --resize_or_crop resize \
               --input_nc 3 \
               --output_nc 3 \
               --label_nc 0 \
               --no_instance \
               --how_many 3061 \
               --which_epoch 40\
                # --phase train \
               

