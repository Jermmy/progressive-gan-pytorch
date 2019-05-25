celeba_hq_dir=/media/liuwq/data/Dataset/Celeba/Celeba-512x512
# celeba_hq_dir=Celeba-128x128

# 1: lsgan+tanh 2: wgangp+tanh
exp=1
if [ $exp == 1 ]; then
    g_lr=1e-3
    d_lr=1e-3
    resolution=32
    epochs=40
    gan_type=lsgan
    norm=pixelnorm
    output_act=tanh
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

rows=4
cols=8

test_epoch=40

result_path=result/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}
ckpt_path=ckpt

phase=test

load_G=${ckpt_path}/G-epoch-40-${resolution}.pkl


python3 train.py --celeba_hq_dir ${celeba_hq_dir} \
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
                --phase ${phase} \
                --ckpt_path ${ckpt_path} \
                --result_path ${result_path} \
                --rows ${rows} \
                --cols ${cols} \
                --load_G ${load_G} 
