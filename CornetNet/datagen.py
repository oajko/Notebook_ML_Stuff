import tensorflow as tf
import os
import json
import numpy as np
import pandas as pd
import glob

"""
Inputs are raw images. And predicting 3 heatmaps for each corner
Have to preprocess for:

- Embedding - nxn map for y = 1
- heatmap - location of corner, nxnxc each corner, c = classes num
- offset - corner accuracy, 

Probably not most efficient or clean datagen, but wanted to experiment with data.Dataset API.
"""

class DataFlow:
    def __init__(self, xpath, y = False):
        self.xpath = xpath
        self.y = y
    
    def gen_data(self):
        file_paths = glob.glob(os.path.join(self.xpath, '*'))
        y_file = [i for i in file_paths if 'coco' in i]
        im_path = [i for i in file_paths if 'coco' not in i]

        file_list = tf.data.Dataset.from_tensor_slices(im_path)
        im_data = file_list.map(self.im_read, num_parallel_calls = tf.data.AUTOTUNE)
        im_data = im_data.prefetch(tf.data.AUTOTUNE)
        if self.y is True:
            with open(y_file[0], mode = 'r', encoding = 'utf-8') as file:
                json_data = json.load(file)
                
            scaled_json = self._y_preprocess(json_data)
            file_list = tf.data.Dataset.from_tensor_slices(im_path)
            file_list = file_list.map(lambda x: tf.strings.as_string(x))

            combined = file_list.map(lambda x: tf.py_function(
                    func = lambda p: self.total(p.numpy().decode(), scaled_json), inp = [x], Tout = tf.float32), num_parallel_calls = tf.data.AUTOTUNE
                )
            
            dataset = tf.data.Dataset.zip((im_data, combined))
            return dataset
        return im_data.map(lambda x: (x,))

    def total(self, _file_name, json_data):
        heatmapl, heatmapr = self.heatmap_preprocess(_file_name, json_data)
        embeddingl, embeddingr = self.embedding_preprocess(_file_name, json_data)
        offsetl, offsetr = self.offset_preprocess(_file_name, json_data)
        x = tf.concat([heatmapl, embeddingl, offsetl, heatmapr, embeddingr, offsetr], axis = -1)
        return x
    
    def offset_preprocess(self, _file_name, json_data):
        tl_tensor = tf.Variable(tf.zeros((128, 128, 2)))
        br_tensor = tf.Variable(tf.zeros((128, 128, 2)))

        im_df = pd.DataFrame(json_data['images'])
        im_id = im_df.loc[im_df.file_name == _file_name.split('\\')[-1], 'id'].values[0]
        anot_df = pd.DataFrame(json_data['annotations'])
        local_json = anot_df.loc[anot_df.image_id == im_id]['bbox']
        for box in local_json.values:
            lx, ty = box[0], box[1]
            rx = lx + box[2]
            by = ty + box[3]

            raw_lx_ds = lx / 2
            int_lx_ds = lx // 2
            raw_ty_ds = ty / 2
            int_ty_ds = ty // 2
            tlx_offsample = raw_lx_ds - int_lx_ds
            tty_offsample = raw_ty_ds - int_ty_ds

            raw_by_ds = by / 2
            int_by_ds = by // 2
            raw_rx_ds = rx / 2
            int_rx_ds = rx // 2
            brx_offsample = raw_rx_ds - int_rx_ds
            bby_offsample = raw_by_ds - int_by_ds

            tl_tensor[int_ty_ds, int_lx_ds, 0].assign(tlx_offsample)
            tl_tensor[int_ty_ds, int_lx_ds, 1].assign(tty_offsample)
            br_tensor[int_by_ds, int_rx_ds, 0].assign(brx_offsample)
            br_tensor[int_by_ds, int_rx_ds, 1].assign(bby_offsample)
        return tf.convert_to_tensor(tl_tensor), tf.convert_to_tensor(br_tensor)
    
    def heatmap_preprocess(self, _file_name, json_data):
        GAUSSIAN_NOISE = self._gaussian_noise()
        tl_tensor = tf.Variable(tf.zeros((128, 128, 8)))
        br_tensor = tf.Variable(tf.zeros((128, 128, 8)))
        
        im_df = pd.DataFrame(json_data['images'])
        anot_df = pd.DataFrame(json_data['annotations'])
        im_id = im_df.loc[im_df.file_name == _file_name.split('\\')[-1], 'id'].values[0]
        local_json = anot_df.loc[anot_df.image_id == im_id]

        for _, local in local_json.iterrows():
            z_idx = local.category_id
            lx, ty, w, h = local.bbox
            rx = lx + w
            by = ty - h

            tl_gaussian_noise = GAUSSIAN_NOISE[abs(min(0, ty - 2)): 5 + 127 - max(127, ty + 2), abs(min(0, lx - 2)): 5 + 127 - max(127, lx + 2)]
            for row in range(tl_gaussian_noise.shape[0]):
                for col in range(tl_gaussian_noise.shape[1]):
                    row_idx = np.clip(ty - 2, 0 , 127) + row
                    col_idx = np.clip(lx - 2, 0, 127) + col
                    tl_tensor[row_idx, col_idx, z_idx].assign(tl_gaussian_noise[row][col])

            br_gaussian_noise = GAUSSIAN_NOISE[abs(min(0, by - 2)): 5 + 127 - max(127, by + 2), abs(min(0, rx - 2)): 5 + 127 - max(127, rx + 2)]
            for row in range(br_gaussian_noise.shape[0]):
                for col in range(br_gaussian_noise.shape[1]):
                    row_idx = np.clip(by - 2, 0 , 127) + row
                    col_idx = np.clip(rx - 2, 0, 127) + col
                    br_tensor[row_idx, col_idx, z_idx].assign(br_gaussian_noise[row][col])

        return tf.convert_to_tensor(tl_tensor), tf.convert_to_tensor(br_tensor)
    
    def embedding_preprocess(self, _file_name, json_data):
        tl_tensor = tf.Variable(tf.zeros((128, 128, 1)))
        br_tensor = tf.Variable(tf.zeros((128, 128, 1)))
        im_df = pd.DataFrame(json_data['images'])
        im_id = im_df.loc[im_df.file_name == _file_name.split('\\')[-1], 'id'].values[0]
        anot_df = pd.DataFrame(json_data['annotations'])
        local_json = anot_df.loc[anot_df.image_id == im_id]['bbox']
        for box in local_json:
            lx, ty = box[0], box[1]
            rx = lx + box[2] 
            by = ty + box[3]
            tl_tensor[ty, lx].assign(np.random.normal())
            br_tensor[by, rx].assign(np.random.normal())
        return tf.convert_to_tensor(tl_tensor), tf.convert_to_tensor(br_tensor)

    def _gaussian_noise(self, radius = 2, sigma = 2 / 3):
        x, y = tf.meshgrid(tf.range(-radius, radius + 1), tf.range(-radius, radius + 1))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        exp_func = tf.exp(- (x * x + y * y) / (2 * sigma * sigma))
        # Norm to 0-1 scale
        return exp_func / tf.reduce_max(exp_func)
    
    # Scale images x, y, w, h to 256 x 256 for gt preprocessing.
    def _y_preprocess(self, json_data):
        im_df = pd.DataFrame(json_data['images'])
        im_df['scale'] = 128 / im_df[['height', 'width']].max(axis = 1).values

        anot_df = pd.DataFrame(json_data['annotations']).rename(columns = {'id': 't_id'})
        anot_df = anot_df.merge(im_df[['id', 'scale']], how = 'left', left_on = 'image_id', right_on = 'id')
        anot_df['bbox'] = anot_df.apply(lambda row: [int(row['bbox'][0] * row['scale']),
            int(row['bbox'][1] * row['scale']),
            int(row['bbox'][2] * row['scale']),
            int(row['bbox'][3] * row['scale'])], axis=1)
        anot_df = anot_df.drop('scale', axis = 1)
        
        im_json = im_df.to_dict(orient = 'records')
        anot_json = anot_df.to_dict(orient = 'records')
        return {'images': im_json, 'annotations': anot_json}

    def im_read(self, im):
        im = tf.io.read_file(im)
        im = tf.io.decode_jpeg(im, 3)
        im = tf.cast(im, tf.float32)
        shape = tf.shape(im)
        height = shape[0]
        width = shape[1]
        shape_max = tf.maximum(height, width)

        if shape_max < 512:
            left_pad = (512 - width) // 2
            right_pad = 512 - width - left_pad * 2
            top_pad = (512 - height) // 2
            bot_pad = 512 - height - top_pad * 2
            padding_ = tf.stack([[top_pad, top_pad + bot_pad], [left_pad, left_pad + right_pad], [0, 0]])
            im = tf.cast(tf.pad(im, padding_), tf.float32)

        elif shape_max > 512:
            scale = 512 / shape_max

            new_height = tf.cast(tf.cast(height, tf.float64) * scale, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float64) * scale, tf.int32)
            resized_image = tf.image.resize(im, [new_height, new_width])

            pad_height = 512 - new_height
            pad_width = 512 - new_width
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            top_pad = pad_height // 2
            bot_pad = pad_height - top_pad

            padding_ = tf.stack([[top_pad, bot_pad], [left_pad, right_pad], [0, 0]])
            im = tf.pad(resized_image, padding_)
        im = tf.cast(im, tf.float32) / 255.0

        return im

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, data, batch, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch
        self.dataset = data.gen_data().batch(batch).prefetch(tf.data.AUTOTUNE)
    
    def __len__(self):
        return len(self.dataset)

    def on_epoch_end(self):
        self.dataset = self.dataset.shuffle(buffer_size=1000)

    def __getitem__(self, indices):
        batch_data = list(self.dataset.skip(indices).take(1).as_numpy_iterator())[0]
        return batch_data