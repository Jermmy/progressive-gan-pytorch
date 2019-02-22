celeba_hq_dir=/media/liuwq/data/Dataset/Celeba/Celeba-128x128
lr=1e-3
batch_size=16
epochs=20
gan_type=lsgan
l_gp=10.
resolution=8
prev_reso=4
alpha=0.5
norm=pixelnorm
output_act=linear

start_idx=0

phase=train

test_epoch=40
load_alpha=1.0

ckpt_path=ckpt/reso-${resolution}x${resolution}/lr_${lr}_${gan_type}_alpha_${alpha}_${norm}_${output_act}
result_path=result/reso-${resolution}x${resolution}/lr_${lr}_${gan_type}_alpha_${alpha}_${norm}_${output_act}

load_G=ckpt/reso-${prev_reso}x${prev_reso}/lr_${lr}_${gan_type}_alpha_${load_alpha}_${norm}_${output_act}/G-epoch-${test_epoch}.pkl
load_D=ckpt/reso-${prev_reso}x${prev_reso}/lr_${lr}_${gan_type}_alpha_${load_alpha}_${norm}_${output_act}/D-epoch-${test_epoch}.pkl

if [ $phase == "test" ]; then
    load_G=${ckpt_path}/G-epoch-${test_epoch}.pkl
fi

python train.py --celeba_hq_dir ${celeba_hq_dir} --lr ${lr} --batch_size ${batch_size} \
                --epochs ${epochs} --gan_type ${gan_type} --l_gp ${l_gp} \
                --resolution ${resolution} \
                --alpha ${alpha} \
                --norm ${norm} \
                --output_act ${output_act} \
                --start_idx ${start_idx} \
                --test_epoch ${test_epoch} \
                --phase ${phase} \
                --ckpt_path ${ckpt_path} \
                --result_path ${result_path} \
                --load_G ${load_G} \
                --load_D ${load_D}
