#$ -cwd
#$ -q gpu.q@@2080 -l gpu=1
#$ -l h_rt=24:00:00,mem_free=32G
#$ -N awd-en8-ex 
#$ -m bea
#$ -j y

module load cuda10.0/toolkit
module load cudnn/7.5.0_cuda10.0
module load nccl/2.4.2_cuda10.0

python -u extract_features.py \
	--nlayers 3 --emsize 400 --nhid 1840 \
	--corpus /exp/dmueller/scale19/lms/awd-lstm/enwik8/corpus.data \
	--resume /exp/dmueller/scale19/lms/awd-lstm/enwik8/ENWIK8.pt \
	--sentences-file /exp/dmueller/scale19/lm-features/eng-ontonotes/train.bytes.txt \
	--vector-outs /exp/dmueller/scale19/lm-features/eng-ontonotes/awd-lstm-enwik8/bytes/train.fwd.features.npy

python -u extract_features.py \
	--nlayers 3 --emsize 400 --nhid 1840 \
	--corpus /exp/dmueller/scale19/lms/awd-lstm/enwik8/corpus.data \
	--resume /exp/dmueller/scale19/lms/awd-lstm/enwik8/ENWIK8.pt \
	--sentences-file /exp/dmueller/scale19/lm-features/eng-ontonotes/testa.bytes.txt \
	--vector-outs /exp/dmueller/scale19/lm-features/eng-ontonotes/awd-lstm-enwik8/bytes/testa.fwd.features.npy
