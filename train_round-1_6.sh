python triplet-train.py --batch_size 256 --img_size 256 --device cuda:0 --data_root /mnt/nvme/data/chromosome --img_path neg-chunk --anno_paths neg-chunk.csv --lr 1e-7 --weight_decay 1e-6 --margin 1. --scheduler_step 800 --scheduler_gamma 0.1 --checkpoint ./models/EmbeddingNet-5.pth --round_id 1 --train_id 3 --epoches 1000 --amp_opt O1 --ohem Semihard