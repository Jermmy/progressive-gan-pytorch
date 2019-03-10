# celeba_hq_dir=/media/liuwq/data/Dataset/Celeba/Celeba-128x128
celeba_hq_dir=Celeba-128x128

# 1: lsgan+linear  2: lsgan+tanh  3: wgangp+linear  4: wgangp+tanh
exp=2
if [ $exp == 1 ]; then
    lr=1e-4
    resolution=32
    epochs=80
    gan_type=lsgan
    norm=pixelnorm
    output_act=linear
    start_idx=0
    test_epoch=40
    load_reso=32
    load_phase=stabilize
    l_gp=1.
elif [ $exp == 2 ]; then
    lr=1e-3
    resolution=8
    epochs=80
    gan_type=lsgan
    norm=pixelnorm
    output_act=tanh
    start_idx=0
    test_epoch=40
    load_reso=4
    load_phase=stabilize
    l_gp=1.
elif [ $exp == 3 ]; then
    lr=1e-4
    resolution=16
    epochs=80
    gan_type=wgangp
    l_gp=5.
    norm=pixelnorm
    output_act=linear
    start_idx=40
    test_epoch=40
    load_phase=stabilize
    load_reso=16
elif [ $exp == 4 ]; then
    lr=1e-4
    resolution=4
    epochs=40
    gan_type=wgangp
    l_gp=1.
    norm=pixelnorm
    output_act=tanh
    start_idx=0
    test_epoch=40
    load_phase=stabilize
    load_reso=4
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


# phase=stabilize
phase=fadein
# phase=test

ckpt_path=ckpt/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}
result_path=result/reso-${resolution}x${resolution}/${phase}_${gan_type}_${norm}_${output_act}

load_G=ckpt/reso-${load_reso}x${load_reso}/${load_phase}_${gan_type}_${norm}_${output_act}/G-epoch-${test_epoch}.pkl
load_D=ckpt/reso-${load_reso}x${load_reso}/${load_phase}_${gan_type}_${norm}_${output_act}/D-epoch-${test_epoch}.pkl

if [ $phase == "test" ]; then
    load_phase=stabilize
    load_G=${ckpt_path}/G-epoch-${test_epoch}.pkl
fi

python train.py --celeba_hq_dir ${celeba_hq_dir} --lr ${lr} --batch_size ${batch_size} \
                --epochs ${epochs} --gan_type ${gan_type} --l_gp ${l_gp} \
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
