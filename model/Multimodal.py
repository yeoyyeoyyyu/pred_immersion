import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization, Activation, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from keras.losses import CategoricalCrossentropy
from keras.metrics import Recall, Precision

from sklearn.model_selection import train_test_split

class Multimodal:
    def __init__(self):
        self.config = {
            'IMG_SIZE': (224, 224, 3),
            'EPOCHS':100,
            'LEARNING_RATE':1e-4,
            'BATCH_SIZE':16,
            'SEED':41,
            'N_CLASS': 5
        }


    def load_data(self, root_dir):
        data_format = {'.jpg':[], '.json':[]}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if 'ipynb' in root: continue
                path = os.path.join(root, file)
                cls_cd = int(root.split('/')[-1][-1])-1
                user_id = root.split('/')[-2]
                first, last = os.path.splitext(file)
                idx = int(first.split('_')[-1])
                data_format[last].append((user_id, cls_cd, idx, path))
        
        df_image = pd.DataFrame(data_format['.jpg'], columns=['USER_ID', 'CLS_CD', 'idx', 'path'])
        df_json = pd.DataFrame(data_format['.json'], columns=['USER_ID', 'CLS_CD', 'idx', 'path'])
        df = pd.merge(df_image, df_json, on=["USER_ID", 'CLS_CD', "idx"], how="inner")
        df.columns = ['USER_ID', 'CLS_CD', 'idx', 'image_path', 'json_path']
        return df


    def json_parsing(self, df):
        def get_data(json_path):
            with open(json_path, 'r') as f:
                jsonData = json.load(f)
            return (jsonData['이미지']['face_box'],
                    jsonData['소음']['decibel'],
                    jsonData['뇌파']['brain_alpha'],
                    jsonData['뇌파']['brain_beta'],
                    jsonData['뇌파']['brain_theta'],
                    jsonData['뇌파']['brain_delta'],
                    jsonData['뇌파']['brain_gamma'])
        
        bBox, decibel = [], []
        alpha, beta, theta, delta, gamma = [], [], [], [], []
        for jsonPath in tqdm(df['json_path'].values, total=len(df)):
            bb, db, a, b, t, d, g = get_data(jsonPath)
            bBox.append(bb)
            decibel.append(db)
            alpha.append(a)
            beta.append(b)
            theta.append(t)
            delta.append(d)
            gamma.append(g)

        df['bBox'] = bBox
        df['begin'] = df['bBox'].apply(lambda x: [x[0][1], x[0][0], 0])
        df['size'] = df['bBox'].apply(lambda x: [x[1][1]-x[0][1], x[1][0]-x[0][0], 3])
        df['decibel'] = decibel
        df['alpha'] = alpha
        df['beta']  = beta
        df['theta'] = theta
        df['delta'] = delta
        df['gamma'] = gamma

        return df

    
    def data_flatten(self, df):
        tot = []
        df = df.drop(columns=['json_path', 'bBox', 'idx'], axis=1)
        for USER_ID, CLS_CD, image_path, begin, size, decibel, alpha, beta, theta, delta, gamma in df.values:
            tmp = []
            tmp.append(USER_ID)
            tmp.append(CLS_CD)
            tmp.append(image_path)
            tmp.append(begin)
            tmp.append(size)
            tmp.append(decibel)
            tmp.extend(alpha)
            tmp.extend(beta)
            tmp.extend(theta)
            tmp.extend(delta)
            tmp.extend(gamma)
            tot.append(tmp)

        return pd.DataFrame(tot, columns=['USER_ID', 'CLS_CD', 'path', 'begin', 'size', 'decibel']+list(range(80)))


    def data_split(self, df):
        df_train, df_valid = train_test_split(df, test_size=0.2, random_state=self.config['SEED'])
        df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=self.config['SEED'])
        self.df_train, self.df_valid = df_train, df_valid


    def preprocessing(self, path, begin, size, decibel, eeg, label, isTrain=False):
        label = tf.one_hot(label, 5)
        
        bin = tf.io.read_file(path)
        image = tf.io.decode_jpeg(bin, channels=3)
        
        # 슬라이싱 실행
        image = tf.slice(image, 
                        begin, 
                        size)
        image = tf.image.resize(image, (224, 224))
        
        if isTrain:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            image = tf.image.random_hue(image, max_delta=0.05)

        image = tf.squeeze(image)
        return (image, decibel, eeg), label


    def generator_train(self):
        for item in self.df_train.values:
            # image_path, begin, size, decibel, eeg, label
            yield item[2], item[3], item[4], item[5], item[6:], item[1]

    
    def generator_valid(self):
        for item in self.df_valid.values:
            # image_path, begin, size, decibel, eeg, label
            yield item[2], item[3], item[4], item[5], item[6:], item[1]


    def generate_dataset(self):
        dataset_train = tf.data.Dataset.from_generator(
            self.generator_train,
            (tf.string, tf.int32, tf.int32, tf.float32, tf.float32, tf.int32),
            ((), (3,), (3,), (10,), (80,), ())
            )

        dataset_valid = tf.data.Dataset.from_generator(
            self.generator_valid,
            (tf.string, tf.int32, tf.int32, tf.float32, tf.float32, tf.int32),
            ((), (3,), (3,), (10,), (80,), ())
            )

        dt = dataset_train.map(lambda *x:self.preprocessing(*x, True), 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dt = dt.batch(self.config['BATCH_SIZE']).prefetch(3)
        dv = dataset_valid.map(self.preprocessing, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dv = dv.batch(self.config['BATCH_SIZE']).prefetch(3)
        return dt, dv


    def get_model(self):
        def fc_blocks(x, channels, act='relu', ratio=0.2):
            x = Dense(channels)(x)
            x = BatchNormalization()(x)
            x = Activation(act)(x)
            x = Dropout(ratio)(x)
            return x

        backbone = MobileNetV3Large(include_top=False, 
                       input_shape=self.config['IMG_SIZE'],
                       weights='imagenet')
    
        input1 = Input(shape=self.config['IMG_SIZE'], dtype=tf.float32)
        input2 = Input(shape=(10,), dtype=tf.float32)
        input3 = Input(shape=(80,), dtype=tf.float32)
        
        x1 = preprocess_input(input1)
        x1 = backbone(x1)
        x1 = GlobalAveragePooling2D()(x1)
        x1 = fc_blocks(x1, 128, 'relu', 0.2)
        x1 = fc_blocks(x1, 32, 'relu', 0.2)

        x2 = fc_blocks(input2, 32, 'relu', 0.2)
        
        m = tf.reduce_min(input3)
        M = tf.reduce_max(input3)
        x3 = (input3 - m) / (M - m)
        x3 = fc_blocks(x3, 64, 'tanh', 0.2)
        x3 = fc_blocks(x3, 128, 'tanh', 0.2)
        x3 = fc_blocks(x3, 32, 'tanh', 0.2)

        x = concatenate([x1, x2, x3])
        x = fc_blocks(x, 8, 'relu', 0.2)
        
        output = Dense(5, activation='softmax')(x)
        model = Model(inputs=[input1, input2, input3], outputs=output)
        return backbone, model


    def train(self, backbone, model, dt, dv):
        backbone.trainable = False
        es = EarlyStopping(monitor='val_loss', 
                        patience=10)
        mc = ModelCheckpoint('/app/train/model/Multimodal/best.h5', 
                            monitor='val_loss', 
                            mode='min', 
                            verbose=1, 
                            save_best_only=True)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['LEARNING_RATE']), 
                    loss=[CategoricalCrossentropy(from_logits=False)], 
                    metrics=[Recall(), Precision()])
        model.fit(dt, epochs=self.config['EPOCHS'], validation_data=dv, callbacks=[es, mc])
        model.save('/app/train/model/Multimodal/last.h5')


if __name__=="__main__":
    pred = Multimodal()
    
    print('전체 데이터 로드')
    df = pred.load_data('/app/data/Multimodal_data')
    print('전체 데이터 수:', len(df), '\n')

    print('학습, 검증 데이터 추출')
    df = pred.json_parsing(df)
    df = pred.data_flatten(df)
    pred.data_split(df)
    print(f'학습 데이터 수: {len(pred.df_train)}, 검증 데이터 수: {len(pred.df_valid)}', '\n')
    dt, dv = pred.generate_dataset()
    
    print('학습 시작')
    backbone, model = pred.get_model()
    pred.train(backbone, model, dt, dv)