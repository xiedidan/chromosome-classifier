python triplet-train.py --checkpoint models/EmbeddingNet-2-11.pth --batch_size 128 --img_size 256 --device cuda:0 --data_root /home/xd/data/chromosome --img_path neg-chunk --anno_paths anno_round-1.csv anno_round-2.csv --lr 1e-6 --weight_decay 1e-5 --margin 1. --scheduler_step 800 --scheduler_gamma 0.1 --round_id 2 --train_id 12 --epoches 1200 --amp_opt O0 --ohem Random
