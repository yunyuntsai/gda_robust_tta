export PYTHONPATH=$PYTHONPATH:$(pwd)

declare -a CorruptionArray=(
                  "fog"  "contrast" "pixelate" "jpeg_compression" "elastic_transform")


for c in ${CorruptionArray[@]}
do              
    CUDA_VISIBLE_DEVICES=0 python main.py --severity 5 --corruption $c --guided_type marginal
done
