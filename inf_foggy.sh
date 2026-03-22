RELIGHT=foggy_1
DATE=exp_1_10_1_v2
SEED_FILE=outputs/exp_10_17_foggy_1.txt # NOTE: hardcode as old one
NGPUS=4

for g in $(seq 0 $((NGPUS-1))); do
  awk -v g=$g -v n=$NGPUS 'NF && (NR-1)%n==g {print $1}' "$SEED_FILE" \
  | while read -r s; do
      python inference.py --base_config configs/base_10_2.yaml --exp_config configs/${DATE}.yaml \
        --relight_type $RELIGHT --gpu $g --seed_offset "$s" --num_seeds 1
    done &
done
wait