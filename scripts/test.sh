################################ Testing ################################
DATA_PATH=YOUR_DATA_PATH
MODEL_NAME=THER2RGB

python test.py --name $MODEL_NAME \
               --dataroot $DATA_PATH \
               --gpu_ids 0 \
               --load_size 256 \
               --nThreads 1\
               --no_flip \
               --serial_batches \
               --resize_or_crop resize \
               --which_epoch 150\
               

