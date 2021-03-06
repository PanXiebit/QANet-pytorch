# QANet

Re-implement QANet with PyTorch.

## Usage

**preprocess data**
python3 data_loader/squad_data.py

**train**
python3 main.py --with_cuda --batch_size 16 --multi_gpu --use_ema

## Experiment

### config
**hyperparameters:**
- dropout_c: 0.1
- dropout_w: 0.05
- d_model: 128
- learning rate: 0.001, using warm-up scheduler
- num_heads: 8
- beta1: 0.8
- beta2: 0.999

**other parameters**
- word_embedding: glove.840B.300d
- char_embedding: glove.840B.300d-char
- grad_clip: 5.0
- para_limit: 400
- ques_limit: 20
- ans_limit: 30
- char_limit: 16

### SQuAD Dataset
||train|dev|step/train_epoch|
|--|--|--|---|
|V1.1|87360|10496|5460|
|v2.0|-|-|-|

|Experimenter|git SHA|Background Search Method|Model|F1|EM|Notes|examples/seconds|
|--|--|--|--|---|---|--|--|
|PanXie|a0c87ba|base model|QANet|78.52|69.13|static PosEnocder, patience 30|35/s|
|PanXie|ff39d3a|without ema|QANet|75.29|64.38|static PosEnocder, patience 19|35/s
|PanXie|7912256|head=1|QANet|77.10|66.91|static PosEnocder, patience 25|35/s|



## Reference
- BangLiu/QANet-PyTorch: https://github.com/BangLiu/QANet-PyTorch
- NLPLearn/QANet: https://github.com/NLPLearn/QANet

