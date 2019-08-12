python3 -m identification.train --wider_train ~/datasets/wider/wider_face_train_bbx_gt.txt --wider_train_prefix ~/datasets/wider/WIDER_train/images \
--wider_val ~/datasets/wider/wider_face_val_bbx_gt.txt --wider_val_prefix ~/datasets/wider/WIDER_val/images \
--depth 50 --epochs 30 --batch_size 1 --model_name wider1
