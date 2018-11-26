#!/usr/bin/python
# coding:utf-8
import os
import random
import torch
import math
import logging
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import parser
from data_loader.squad_data import prepro, get_loader
from utils.file_utils import pickle_load_large_file
from QANet import QANet
from trainer.ema import EMA
from trainer.qanet_train import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('qanet.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]   >> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = get_logger()

args = parser.parse_args()

def main(args=args, logger=logger):
    random_seed = None

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")

    # process word vectors and datasets
    if not args.processed_data:
        prepro(args)

    # load word vectors
    wv_tensor = torch.FloatTensor(
        np.array(pickle_load_large_file(args.word_emb_file), dtype=np.float32))
    cv_tensor = torch.FloatTensor(
        np.array(pickle_load_large_file(args.char_emb_file), dtype=np.float32))
    wv_word2ix = pickle_load_large_file(args.word_dictionary)
    print(wv_tensor.shape, cv_tensor.shape, len(wv_word2ix))

    # load datasets
    train_dataloader = get_loader(
        args.train_examples_file, args.batch_size, shuffle=True)
    dev_dataloader = get_loader(
        args.dev_examples_file, args.batch_size, shuffle=True)
    print(len(train_dataloader), len(dev_dataloader))

    # load model
    model = QANet(
        wv_tensor,
        cv_tensor,
        args.para_limit,
        args.ques_limit,
        args.d_model,
        num_head=args.num_head,
        train_cemb=(not args.pretrained_char),
        pad=wv_word2ix["<PAD>"]
    )
    model.summary()

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    print(model)
    model.to(device)

    # exponential moving average
    ema = EMA(args.decay)
    if args.use_ema:
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    # set optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        params=parameters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=3e-7)
    cr = 1.0 / math.log(args.lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log(ee + 1)
        if ee < args.lr_warm_up_num else 1)

    # set loss, metrics
    loss = torch.nn.CrossEntropyLoss()

    identifier = type(model).__name__ + '_'
    trainer = Trainer(
        args, model, loss,
        train_data_loader=train_dataloader,
        dev_data_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        identifier=identifier,
        logger=logger
    )

    start = datetime.now()
    trainer.train()
    print("Time of training model ", datetime.now() - start)


if __name__ == "__main__":
    main()
