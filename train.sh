python train.py \
    --image_dir ./ourpets-photos \
    --model_dir ./my-net \
    --output_graph ./my-net/ourpets-1.pb \
    --output_labels ./my-net/ourpets-1_labels.txt \
    --how_many_training_steps 50 \
    --bottleneck_dir ./bottleneck \
    --flip_left_right \
    -- \
    --random_crop 10 --random_scale 10 --random_brightness 10 
