#celeba_hq_dir=/media/liuwq/data/Dataset/Celeba/Celeba-128x128
celeba_hq_dir=data/Celeba-128x128
lr=1e-3
batch_size=16
epochs=40
gan_type=lsgan
l_gp=10.
resolution=4
alpha=1.0
norm=pixelnorm
output_act=tanh

start_idx=0

ckpt_path=ckpt/reso-${resolution}x${resolution}/lr_${lr}_${gan_type}_alpha_${alpha}_${norm}_${output_act}
result_path=result/reso-${resolution}x${resolution}/lr_${lr}_${gan_type}_alpha_${alpha}_${norm}_${output_act}

# load_G=ckpt/reso-4x4/lr_${lr}_${gan_type}_alpha_${alpha}/G-epoch-10.pkl
# load_D=ckpt/reso-4x4/lr_${lr}_${gan_type}_alpha_${alpha}/D-epoch-10.pkl

python3 train.py --celeba_hq_dir ${celeba_hq_dir} --lr ${lr} --batch_size ${batch_size} \
                --epochs ${epochs} --gan_type ${gan_type} --l_gp ${l_gp} \
                --resolution ${resolution} \
                --alpha ${alpha} \
                --norm ${norm} \
                --output_act ${output_act} \
                --start_idx ${start_idx} \
                --ckpt_path ${ckpt_path} --result_path ${result_path}