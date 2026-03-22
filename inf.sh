########## Human Relight ##########
# python inference_outpaint.py \
#     --base_config configs/base_10_2.yaml \
#     --exp_config configs/exp_1_10_1_v2.yaml \
#     --relight_type golden_sunlight_1 \
#     --gpu 0 \
#     --seed_offset 0 \
#     --num_seeds 1

# python inference_t2i.py \
#   --base_config configs/base_10_2.yaml \
#   --exp_config configs/exp_1_10_1_v2.yaml \
#   --relight_type golden_sunlight_1 \
#   --gpu 0 \
#   --seed_offset 0 \
#   --num_seeds 1

########## Prepare Data into img2img-turbo format ##########
# python prepare_data.py \
#   --base_config configs/base_10_2.yaml \
#   --exp_config configs/2_24_drive_v2.yaml \
#   --relight_type golden_sunlight_1 \
#   --gpu 7