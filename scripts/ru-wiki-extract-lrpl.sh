#$ -cwd
#$ -q gpu.q@@2080 -l gpu=1
#$ -l h_rt=24:00:00,mem_free=32G
#$ -N awd-ruwik-ex
#$ -m bea
#$ -j y

module load cuda10.0/toolkit
module load cudnn/7.5.0_cuda10.0
module load nccl/2.4.2_cuda10.0

python -u extract_features.py \
	--nlayers 3 --emsize 400 --nhid 1840 \
	--corpus /exp/dmueller/scale19/lms/awd-lstm/ru-wiki/corpus.data \
	--resume /exp/dmueller/scale19/lms/awd-lstm/ru-wiki/cur_best.pt \
	--sentences-file /exp/dmueller/scale19/lm-features/rus-lrpl/train.bytes.txt \
	--vector-outs /exp/dmueller/scale19/lm-features/rus-lrpl/awd-lstm-ruwiki/bytes/train.fwd.features.npy

python -u extract_features.py \
	--nlayers 3 --emsize 400 --nhid 1840 \
	--corpus /exp/dmueller/scale19/lms/awd-lstm/ru-wiki/corpus.data \
	--resume /exp/dmueller/scale19/lms/awd-lstm/ru-wiki/cur_best.pt \
	--sentences-file /exp/dmueller/scale19/lm-features/rus-lrpl/testa.bytes.txt \
	--vector-outs /exp/dmueller/scale19/lm-features/rus-lrpl/awd-lstm-ruwiki/bytes/testa.fwd.features.npy
