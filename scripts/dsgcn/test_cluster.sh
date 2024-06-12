config=./dsgcn/configs/cfg_test.py
load_from=./data/pretrained_models/pretrained_gcn.pth

export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=. python dsgcn/main.py \
    --stage det \
    --phase test \
    --config $config \
    --load_from $load_from \
    --save_output
