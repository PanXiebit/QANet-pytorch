#!/usr/bin/python
# coding:utf-8

import os
import logging
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
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

data_folder = "/home/zhengyinhe/xiepan/squad/data/"
parser = argparse.ArgumentParser(description='Lucy')

# dataset
parser.add_argument(
    '--processed_data',
    default=True, action='store_true',
    help='whether the dataset already processed')
parser.add_argument(
    '--train_file',
    default=data_folder + 'original/squad1.1/train-v1.1.json',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=data_folder + 'original/squad1.1/dev-v1.1.json',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--train_examples_file',
    default=data_folder + 'processed/squad1.1/train-v1.1-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=data_folder + 'processed/squad1.1/dev-v1.1-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--train_meta_file',
    default=data_folder + 'processed/squad1.1/train-v1.1-meta.pkl',
    type=str, help='path of train dataset meta file')
parser.add_argument(
    '--dev_meta_file',
    default=data_folder + 'processed/squad1.1/dev-v1.1-meta.pkl',
    type=str, help='path of dev dataset meta file')
parser.add_argument(
    '--train_eval_file',
    default=data_folder + 'processed/squad1.1/train-v1.1-eval.pkl',
    type=str, help='path of train dataset eval file')
parser.add_argument(
    '--dev_eval_file',
    default=data_folder + 'processed/squad1.1/dev-v1.1-eval.pkl',
    type=str, help='path of dev dataset eval file')
parser.add_argument(
    '--val_num_batches',
    default=500, type=int,
    help='number of batches for evaluation (default: 500)')

# embedding
parser.add_argument(
    '--glove_word_file',
    default=data_folder + 'original/Glove/glove.840B.300d.txt',
    type=str, help='path of word embedding file')
parser.add_argument(
    '--glove_word_size',
    default=int(2.2e6), type=int,
    help='Corpus size for Glove')
parser.add_argument(
    '--glove_dim',
    default=300, type=int,
    help='word embedding size (default: 300)')
parser.add_argument(
    '--word_emb_file',
    default=data_folder + 'processed/squad1.1/word_emb.pkl',
    type=str, help='path of word embedding matrix file')
parser.add_argument(
    '--word_dictionary',
    default=data_folder + 'processed/squad1.1/word_dict.pkl',
    type=str, help='path of word embedding dict file')

parser.add_argument(
    '--pretrained_char',
    default=False, action='store_true',
    help='whether train char embedding or not')
parser.add_argument(
    '--glove_char_file',
    default=data_folder + "original/Glove/glove.840B.300d-char.txt",
    type=str, help='path of char embedding file')
parser.add_argument(
    '--glove_char_size',
    default=94, type=int,
    help='Corpus size for char embedding')
parser.add_argument(
    '--char_dim',
    default=64, type=int,
    help='char embedding size (default: 64)')
parser.add_argument(
    '--char_emb_file',
    default=data_folder + 'processed/squad1.1/char_emb.pkl',
    type=str, help='path of char embedding matrix file')
parser.add_argument(
    '--char_dictionary',
    default=data_folder + 'processed/squad1.1/char_dict.pkl',
    type=str, help='path of char embedding dict file')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=32, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=30, type=int,
    help='number of total epochs (default: 30)')

# debug
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (devault: 2)')

# checkpoint
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2, type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--save_dir',
    default='checkpoints_1/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--print_freq',
    default=200, type=int,
    help='print training information frequency (default: 10 steps)')

# cuda
parser.add_argument(
    '--with_cuda',
    default=False, action='store_true',
    help='use CPU in case there\'s no GPU support')
parser.add_argument(
    '--multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')

# log & visualize
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')

# optimizer & scheduler & weight & exponential moving average
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_scheduler',
    default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument(
    '--use_grad_clip',
    default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument(
    '--grad_clip',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=20, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--para_limit',
    default=400, type=int,
    help='maximum context token number')
parser.add_argument(
    '--ques_limit',
    default=50, type=int,
    help='maximum question token number')
parser.add_argument(
    '--ans_limit',
    default=30, type=int,
    help='maximum answer token number')
parser.add_argument(
    '--char_limit',
    default=16, type=int,
    help='maximum char number in a word')
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='attention num head')
