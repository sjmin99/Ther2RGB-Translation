### Using labels only
DATA_PATH=datasets/MTN/traindata
CHECKPOINTS_PATH=debug
python train.py --gpu_ids 1 --name $CHECKPOINTS_PATH --batch_size 8 --load_size 512 --crop_size 256 --label_nc 0 --input_nc 1 --output_nc 3 --dataroot $DATA_PATH --resize_or_crop scale_width_and_crop --nThreads 16 --no_instance \
                