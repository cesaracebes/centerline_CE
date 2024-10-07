
python train_consistency_losses_2d_patches.py --dataset FIVES_D --save_path retina/FIVES_D/dice          --loss1 dice
python train_consistency_losses_2d_patches.py --dataset FIVES_D --save_path retina/FIVES_D/dice_05cldice --loss1 dice --loss2 cldice --alpha2 0.5
python train_consistency_losses_2d_patches.py --dataset FIVES_D --save_path retina/FIVES_D/dice_clce     --loss1 dice  --loss2 clce --alpha2 1.0

