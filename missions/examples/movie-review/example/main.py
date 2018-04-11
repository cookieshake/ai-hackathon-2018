# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import torch

import nsml
from dataset import MovieReviewDataset, preprocess
from torch.utils.data import DataLoader
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

import pickle


token_list = None
# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        global token_list
        os.makedirs(filename, exist_ok=True)
        model.save_weights(os.path.join(filename, 'model'))
        pickle.dump(token_list, open(os.path.join(filename, 'token_list'), 'wb'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        global token_list

        model.load_weights(os.path.join(filename, 'model'))
        token_list = pickle.load(open(os.path.join(filename, 'token_list'), 'rb'))

        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        global token_list
        input1, input2, _ = preprocess(raw_data, config.strmaxlen, token_list=token_list or 1)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model.predict([input1, input2])
        point = output_prediction.flatten().tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)

from keras.models import Sequential, Model, Input
from keras.layers import Embedding, Flatten, Dense, GRU, concatenate, AlphaDropout, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import LambdaCallback

if __name__ == '__main__':
    print('Start!')
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=8)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    def create_model():
        acti_init = {
            'activation': 'selu',
            'kernel_initializer': 'lecun_normal'
        }
        dropout_rate = 0.5

        input1 = Input(shape=(config.strmaxlen,))

        x1 = Embedding(4096, 64, input_length=config.strmaxlen)(input1)
        x1 = GRU(64, return_sequences=True, dropout=dropout_rate)(x1)
        x1 = GRU(64, return_sequences=True, go_backwards=True, dropout=dropout_rate)(x1)
        x1 = GRU(64, dropout=dropout_rate)(x1)
        x1 = Dense(64, **acti_init)(x1)
        x1 = AlphaDropout(dropout_rate)(x1)

        input2 = Input(shape=(1 + 256 + 4096,))
        x2 = Dense(256, **acti_init)(input2)
        x2 = AlphaDropout(dropout_rate)(x2)
        x2 = Dense(256, **acti_init)(input2)
        x2 = AlphaDropout(dropout_rate)(x2)
        x2 = Dense(64, **acti_init)(x2)
        x2 = AlphaDropout(dropout_rate)(x2)
        
        x = concatenate([x1, x2])
        x = Dense(128, **acti_init)(x)
        x = AlphaDropout(dropout_rate)(x)
        x = Dense(1, activation='sigmoid')(x)
        x = Lambda(lambda x: x * 9 + 1)(x)

        return Model(inputs=[input1, input2], outputs=x)

    model = create_model()
    model.compile(
        loss='mse',
        optimizer=Adam(0.0003),
        metrics=[]
    )
    model.summary()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        print("Now Loading Dataset...")
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        token_list = dataset.token_list
        print("Loading Dataset Done")

        batches_per_epoch = int(len(dataset.labels) / config.batch) + 1

        # 학습을 수행합니다.
        def on_epoch_end(epoch, logs):
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=logs['loss'], step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

        def on_batch_end(batch, logs):
            if batch % 50 == 1 or batch == batches_per_epoch:
                print("Batch: {}/{}, Loss: {}".format(batch, batches_per_epoch, logs['loss']))


        callback = LambdaCallback(on_epoch_end=on_epoch_end, on_batch_end=on_batch_end)
        model.fit(
            x=[dataset.input1, dataset.input2],
            y=dataset.labels, 
            batch_size=config.batch,
            epochs=config.epochs,
            verbose=2,
            callbacks=[callback],
            validation_split=0.1,
            shuffle=True,
        )

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)