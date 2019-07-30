python3 -m recognition.train --casia_list /home/ehp/tmp/datasets/CASIA-maxpy-clean/train.txt --casia_root /home/ehp/tmp/datasets/CASIA-maxpy-clean --lfw_root /home/ehp/tmp/datasets/lfw \
--lfw_pair_list /home/ehp/git/arcface/lfw_test_pair.txt --model_name recongition3 --batch_size 20 --loss adacos --print_freq 20 --depth 50
