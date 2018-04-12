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

import os

import numpy as np
from torch.utils.data import Dataset

from kor_char_parser import decompose_str_as_one_hot


class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')

        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.input1, self.input2, self.token_list = preprocess(f.readlines(), max_length)
        
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = np.array([np.float32(x) for x in f.readlines()]).flatten()

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.input1[idx], self.input2[idx], self.labels[idx]

from kor_char_parser import create_ngram
from collections import Counter, defaultdict
from functools import reduce
from tqdm import tqdm

def preprocess(data: list, max_length: int, token_list=None):
    MAX_TOKEN = 4096

    # input1

    if not token_list:
        counter = Counter()
        for datum in tqdm(data, mininterval=1, bar_format='{r_bar}\n'):
            counter.update(create_ngram(datum, warning=False))
        token_list = [token for token, count in counter.most_common(MAX_TOKEN)]

    token_dict = {token: i for i, token in enumerate(token_list)}
    
    zero_padding1 = np.zeros((len(data), max_length), dtype=np.uint16)
    zero_padding2 = np.zeros((len(data), 1 + 256 + MAX_TOKEN), dtype=np.uint8)

    idx = 0
    for datum in tqdm(data, mininterval=1, bar_format='{r_bar}\n'):
        # input 1
        tokens = [token_dict[token] for token in create_ngram(datum, warning=False) if token in token_dict]

        length = len(tokens)
        if length >= max_length:
            zero_padding1[idx, :max_length] = np.array(tokens)[:max_length]
        else:
            zero_padding1[idx, :length] = np.array(tokens)

        # input2
        zero_padding2[idx, 0] = len(datum)

        counter = Counter(decompose_str_as_one_hot(datum, warning=False))
        counted = np.array([counter[i] for i in range(256)])
        zero_padding2[idx, 1 : 257] = counted
        
        counter = Counter(tokens)
        for token_i, count in counter.items():
            zero_padding2[idx, token_i + 257] = count

        idx += 1


    zero_padding1 = zero_padding1.reshape(len(data), -1)
    zero_padding2 = zero_padding2.reshape(len(data), -1)

    return zero_padding1, zero_padding2, token_list
