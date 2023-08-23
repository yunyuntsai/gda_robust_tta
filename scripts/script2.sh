export PYTHONPATH=$PYTHONPATH:$(pwd)

declare -a CorruptionArray=( "gaussian_noise" "brightness"  "snow"   "frost"   "fog"  "contrast" "pixelate" "jpeg_compression" "elastic_transform")


for c in ${CorruptionArray[@]}
do              
    CUDA_VISIBLE_DEVICES=1 python main.py --severity 3 --corruption $c --guided_type marginal
done

# for c in ${CorruptionArray[@]}
# do              
#     CUDA_VISIBLE_DEVICES=3 python main.py --severity 3 --corruption $c --guided_type marginal --ensemble
# done
