# python litgpt/scripts/prepare_dataset.py \
# --dataset "izumi-lab/wikipedia-ja-20230720" \
# --output_file "./dataset/train/wiki_js_2023.txt"

# python litgpt/scripts/prepare_dataset.py \
# --dataset "izumi-lab/wikinews-ja-20230728" \
# --output_file "./dataset/val/wiki_js_2023.txt"

data_path="./dataset"
split -l  10000 -d --additional-suffix=.txt  $data_path/train/wiki_js_2023.txt $data_path/train_splited/data_

split -l  10000 -d --additional-suffix=.txt   $data_path/val/wiki_js_2023.txt $data_path/val_splited/data_