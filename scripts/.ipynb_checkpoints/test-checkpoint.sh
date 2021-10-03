################################ Testing ################################
# labels only
DATA_PATH=../datasets/Luckydata
# DATA_PATH=datasets/
CHECKPOINTS_PATH=Unet_lucky_T2ab_noise

python test.py --name $CHECKPOINTS_PATH \
               --dataroot $DATA_PATH \
               --gpu_ids 0 \
               --load_size 256 \
               --nThreads 16\
               --no_flip \
               --serial_batches \
               --resize_or_crop resize \
               --input_nc 3 \
               --output_nc 2 \
               --label_nc 0 \
               --no_instance \
               --how_many 3061 \
               --which_epoch 200\
               --netG Unet \
               

