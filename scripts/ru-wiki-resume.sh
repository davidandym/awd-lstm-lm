#$ -cwd
#$ -q gpu.q@@2080 -l gpu=1
#$ -l h_rt=72:00:00,mem_free=32G
#$ -N awd-lstm
#$ -m bea
#$ -j y

module load cuda10.0/toolkit
module load cudnn/7.5.0_cuda10.0
module load nccl/2.4.2_cuda10.0



python -u main.py --epochs 50 --nlayers 3 --emsize 400 --nhid 1840 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.4 --wdrop 0.2 --wdecay 1.2e-6 --bptt 200 --batch_size 128 --optimizer adam --lr 1e-3 --data /exp/dmueller/wiki/ru/splits --save /exp/dmueller/scale19/lms/awd-lstm/ru-wiki/cur_best.pt --when 25 35 --bytes --output-dir /exp/dmueller/scale19/lms/awd-lstm/ru-wiki --resume /exp/dmueller/scale19/lms/awd-lstm/ru-wiki/cur_best.pt
