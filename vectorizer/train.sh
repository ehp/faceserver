#python3 -m identification.train --wider_train /home/ehp/tmp/datasets/wider/sample.txt --wider_train_prefix /home/ehp/tmp/datasets/wider/sample/images \
#--wider_val /home/ehp/tmp/datasets/wider/sample_val.txt --wider_val_prefix /home/ehp/tmp/datasets/wider/sample_val/images \
#--depth 50 --epochs 30 --batch_size 1 --model_name wider_sample1

python3 -m identification.train --wider_train /home/ehp/tmp/datasets/wider/wider_face_train_bbx_gt.txt --wider_train_prefix /home/ehp/tmp/datasets/wider/WIDER_train/images \
--wider_val /home/ehp/tmp/datasets/wider/wider_face_val_bbx_gt.txt --wider_val_prefix /home/ehp/tmp/datasets/wider/WIDER_val/images \
--depth 50 --epochs 30 --batch_size 1 --model_name widernew1
