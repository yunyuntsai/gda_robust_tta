export PYTHONPATH=$PYTHONPATH:$(pwd)

declare -a CorruptionArray=( 'gaussian_noise' 
    'defocus_blur' 'glass_blur' 
    'zoom_blur'  'fog' 'brightness', 
    'contrast' 'pixelate' 'jpeg_compression') # "glass_blur" "defocus_blur" "zoom_blur" "motion_blur" )

# declare -a CorruptionArray=( 
    #  'elastic_transform' 'pixelate' 'jpeg_compression') 

for c in ${CorruptionArray[@]}
do              
    CUDA_VISIBLE_DEVICES=0 python main.py --severity 3 --corruption $c --guided_type supervised --skip_timesteps 45
done

# for c in ${CorruptionArray[@]}
# do              
#     CUDA_VISIBLE_DEVICES=2 python main.py --severity 3 --corruption $c --guided_type marginal 
# done
