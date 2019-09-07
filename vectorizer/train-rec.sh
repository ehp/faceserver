python3 -m recognition.train --casia_list ~/datasets/CASIA-maxpy-clean/train.txt --casia_root ~/datasets/CASIA-maxpy-clean --lfw_root ~/datasets/lfw \
--lfw_pair_list lfw_test_pair.txt --model_name recongition1 --batch_size 20 --loss adacos --print_freq 20 --net resnet50
