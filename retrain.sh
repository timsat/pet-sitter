python retrain.py \
    --image_dir ./ourpets-photos \
    --model_dir ./imagenet \
    --output_graph ./my-net/ourpets.pb \
    --output_labels ./my-net/ourpets_labels.txt \
    --how_many_training_steps 120 \
    --flip_left_right \
    --random_crop 10 --random_scale 10 --random_brightness 10 
