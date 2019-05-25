celeba_hq_dir=/media/liuwq/data/Dataset/Celeba/Celeba-512x512
# celeba_hq_dir=Celeba-128x128

# 1: lsgan+tanh 2: wgangp+tanh
exp=1
if [ $exp == 1 ]; then
    g_lr=1e-3
    d_lr=1e-3
    resolution=128
    epochs=40
    gan_type=lsgan
    norm=pixelnorm
    output_act=tanh
    start_idx=0
    test_epoch=0
    load_reso=512
    load_phase=stabilize
    l_gp=1.
    device_id=1
elif [ $exp == 2 ]; then
    g_lr=1e-5
    d_lr=1e-5
    resolution=16
    epochs=40
    gan_type=wgangp
    l_gp=1
    norm=pixelnorm
    output_act=tanh
    start_idx=0
    test_epoch=36
    load_phase=fadein
    load_reso=16
    device_id=1
fi

if [ $resolution == '256' ]; then
    batch_size=14
elif [ $resolution == '512' ]; then
    batch_size=6
elif [ $resolution == '1024' ]; then
    batch_size=3
else
    batch_size=16
fi

progress=False

phase=stabilize
# phase=fadein
# phase=test

if [ $progress == 'True' ]; then
    ckpt_path=/media/liuwq/data/Dataset/Celeba/ckpt/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}
    result_path=/media/liuwq/data/Dataset/Celeba/result/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}
else
    ckpt_path=/media/liuwq/data/Dataset/Celeba/ckpt/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}_nonpro
    result_path=/media/liuwq/data/Dataset/Celeba/result/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}_nonpro    
fi

load_G=ckpt/reso-${load_reso}x${load_reso}/${load_phase}_${gan_type}_${norm}_${output_act}/G-epoch-${test_epoch}.pkl
load_D=ckpt/reso-${load_reso}x${load_reso}/${load_phase}_${gan_type}_${norm}_${output_act}/D-epoch-${test_epoch}.pkl


if [ $phase == "test" ]; then
    load_phase=stabilize
    load_G=${ckpt_path}/G-epoch-${test_epoch}.pkl
fi

python train.py --celeba_hq_dir ${celeba_hq_dir} \
                --g_lr ${g_lr} \
                --d_lr ${d_lr} \
                --batch_size ${batch_size} \
                --epochs ${epochs} \
                --gan_type ${gan_type} \
                --l_gp ${l_gp} \
                --device_id ${device_id} \
                --resolution ${resolution} \
                --norm ${norm} \
                --output_act ${output_act} \
                --start_idx ${start_idx} \
                --test_epoch ${test_epoch} \
                --phase ${phase} \
                --ckpt_path ${ckpt_path} \
                --result_path ${result_path} \
                --load_G ${load_G} \
                --load_D ${load_D}
