export PYTHONPATH=$PYTHONPATH:$(pwd)

declare -a CorruptionArray=('gaussian_noise' 'defocus_blur' 'glass_blur' 'zoom_blur') 

for c in ${CorruptionArray[@]}
do              
    CUDA_VISIBLE_DEVICES=0 python main.py --severity 3 --corruption $c --guided_type weighted-marginal --skip_timesteps 45
done

#CUDA_VISIBLE_DEVICES=0 python main.py --severity 3 --corruption $c --guided_type marginal --skip_timesteps 45 --- IGNORE ---