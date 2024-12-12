import tensorflow as tf
import pandas as pd
import json
import os
import numpy as np

'''
Aquarium dataset:
- Input image of 256x256
- Output of 64x64
Reduced from 512x512 -> 128x128 in paper to allow local machine running

gt data in json format
Let's agg data in pandas df
'''

class DataFlow():
    def __init__(self, xpath, label_data = False):
        self.xpath = xpath
        self.label_data = label_data
    
    def data_gen(self):
        im_files = [os.path.join(self.xpath, i) for i in os.listdir(self.xpath) if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg')]
        im_dataset = tf.data.Dataset.from_tensor_slices(im_files)
        im_data = im_dataset.map(lambda x: self._image_preprocess(x))
        if self.label_data is True:
            json_file = [i for i in os.listdir(self.xpath) if i.endswith('.json')][0]
            with open (os.path.join(self.xpath, json_file), 'r') as file:
                label_df = self._label_form(json.load(file))
            file_list = im_dataset.map(lambda x: tf.strings.as_string(x))
            combined = file_list.map(lambda x: tf.py_function(
                func = lambda p: self._aggregate(p.numpy().decode(), label_df), inp = [x], Tout = tf.float32), num_parallel_calls = tf.data.AUTOTUNE
            )
            dataset = tf.data.Dataset.zip((im_data, combined))
            return dataset
        return im_data.map(lambda x: (x,))

    def _aggregate(self, im_name, df):
        im_name = im_name.split('\\')[-1]
        temp_df = df.loc[df.file_name == im_name]
        tl_heat, br_heat = self._heatmap(temp_df)
        tl_embed, br_embed = self._embedding(temp_df)
        tl_off, br_off, cen_off = self._offset(temp_df)
        cen_heat = self._center_heatmap(temp_df)
        return tf.concat([tl_heat, tl_embed, tl_off, br_heat, br_embed, br_off, cen_heat, cen_off], axis = -1)    
    
    def _heatmap(self, df):
        tl_heatmap = tf.Variable(initial_value = tf.zeros((64, 64, 8)))
        br_heatmap = tf.Variable(initial_value = tf.zeros((64, 64, 8)))
        gaussian_noise = self._gaussian_noise()
        for _, row_ in df.iterrows():
            lx = row_.x
            ly = row_.y
            rx = row_.x + row_.w
            ry = row_.y + row_.h
            tlgn = gaussian_noise[abs(min(0, ly - 2)): 5 + 63 - max(63, ly + 2), abs(min(0, lx - 2)): 5 + 63 - max(63, lx + 2)]
            for row in range(tlgn.shape[0]):
                for col in range(tlgn.shape[1]):
                    row_idx = np.clip(ly - 2, 0 , 63) + row
                    col_idx = np.clip(lx - 2, 0, 63) + col
                    tl_heatmap[row_idx, col_idx, row_.category_id].assign(tlgn[row][col])
            brgn = gaussian_noise[abs(min(0, ry - 2)): 5 + 63 - max(63, ry + 2), abs(min(0, rx - 2)): 5 + 63 - max(63, rx + 2)]
            for row in range(brgn.shape[0]):
                for col in range(brgn.shape[1]):
                    row_idx = np.clip(ry - 2, 0 , 63) + row
                    col_idx = np.clip(rx - 2, 0, 63) + col
                    br_heatmap[row_idx, col_idx, row_.category_id].assign(brgn[row][col])
        return tf.convert_to_tensor(tl_heatmap), tf.convert_to_tensor(br_heatmap)

    def _center_heatmap(self, df):
        c_heatmap = tf.Variable(initial_value = tf.zeros((64, 64, 8)))
        for _, row_ in df.iterrows():
            center_x = row_.x + row_.w // 2
            center_y = row_.y + row_.h // 2
            gn = self._gaussian_noise()
            gn = gn[abs(min(0, center_y - 2)): 5 + 63 - max(63, center_y + 2), abs(min(0, center_x - 2)): 5 + 63 - max(63, center_x + 2)]
            for row in range(gn.shape[0]):
                for col in range(gn.shape[1]):
                    row_idx = np.clip(center_y - 2, 0 , 63) + row
                    col_idx = np.clip(center_x - 2, 0, 63) + col
                    c_heatmap[row_idx, col_idx, row_.category_id].assign(gn[row][col])
        return tf.convert_to_tensor(c_heatmap)

    def _gaussian_noise(self, radius = 2, sigma = 2 / 3):
        x, y = tf.meshgrid(tf.range(-radius, radius + 1), tf.range(-radius, radius + 1))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        exp_func = tf.exp(- (x * x + y * y) / (2 * sigma * sigma))
        return exp_func / tf.reduce_max(exp_func)

    def _embedding(self, df):
        tl_embed = tf.Variable(initial_value = tf.zeros((64, 64, 1)))
        br_embed = tf.Variable(initial_value = tf.zeros((64, 64, 1)))
        for _, row in df.iterrows():
            tl_embed[row.y, row.x].assign(np.random.normal())
            br_embed[row.y + row.h, row.x + row.w].assign(np.random.normal())
        return tf.convert_to_tensor(tl_embed), tf.convert_to_tensor(br_embed)

    def _offset(self, df):
        tl_embed = tf.Variable(initial_value = tf.zeros((64, 64, 2)))
        br_embed = tf.Variable(initial_value = tf.zeros((64, 64, 2)))
        cen_embed = tf.Variable(initial_value = tf.zeros((64, 64, 2)))
        for _, row in df.iterrows():
            lx = row.x
            ly = row.y
            rx = row.x + row.w
            ry = row.y + row.h
            tl_embed[ly, lx, 0].assign(lx / 4 - lx // 4)
            tl_embed[ly, lx, 1].assign(ly / 4 - ly // 4)
            br_embed[ry, rx, 0].assign(rx / 4 - rx // 4)
            br_embed[ry, rx, 1].assign(ry / 4 - ry // 4)
            cx = (lx + row.w // 2)
            cy = (ly + row.h // 2)
            cen_embed[cy, cx, 0].assign(cx / 4 - cx // 4)
            cen_embed[cy, cx, 1].assign(cy / 4 - cy // 4)
        return tf.convert_to_tensor(tl_embed), tf.convert_to_tensor(br_embed), tf.convert_to_tensor(cen_embed)

    def _label_form(self, labelled_data):
        df = pd.DataFrame(labelled_data['annotations'])[['image_id', 'category_id', 'bbox']]
        temp_df = pd.DataFrame(labelled_data['images'])[['id', 'file_name', 'height', 'width']]
        df = df.merge(temp_df, how = 'left', left_on = 'image_id', right_on = 'id')
        df = pd.concat([df, pd.DataFrame(df['bbox'].to_list(), columns = ['x', 'y', 'w', 'h'])], axis = 1).drop(['bbox', 'id'], axis = 1)
        df['h_scale'] = 64 / df.height
        df['w_scale'] = 64 / df.width
        df['x'] = (df['x'] * df.w_scale).astype(int)
        df['y'] = (df['y'] * df.h_scale).astype(int)
        df['w'] = (df['w'] * df.w_scale).astype(int)
        df['h'] = (df['h'] * df.h_scale).astype(int)
        return df
    
    # Apply padding and Resizing
    def _image_preprocess(self, image):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, 3)
        image = tf.cast(image, tf.float32)
        shape = tf.shape(image)
        width = shape[1]
        height = shape[0]
        max_dim = tf.maximum(height, width)

        if max_dim < 256:
            left_pad = (256 - width) // 2
            right_pad = 256 - width - left_pad * 2
            top_pad = (256 - height) // 2
            bot_pad = 256 - height - top_pad * 2
            padding_ = tf.stack([[top_pad, top_pad + bot_pad], [left_pad, left_pad + right_pad], [0, 0]])
            image = tf.cast(tf.pad(image, padding_), tf.float32)
        elif max_dim > 256:
            scale = 256 / max_dim
            new_height = tf.cast(tf.cast(height, tf.float64) * scale, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float64) * scale, tf.int32)
            resized_image = tf.image.resize(image, [new_height, new_width])
            pad_height = 256 - new_height
            pad_width = 256 - new_width
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            top_pad = pad_height // 2
            bot_pad = pad_height - top_pad
            padding_ = tf.stack([[top_pad, bot_pad], [left_pad, right_pad], [0, 0]])
            image = tf.pad(resized_image, padding_)
        image = tf.cast(image, tf.float32) / 255.0
        return image

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch, **kwargs):
        super().__init__(**kwargs)
        self.dataset = data.shuffle(buffer_size = 1000).batch(batch).prefetch(tf.data.AUTOTUNE)
        self.iterator = iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            batch = next(self.iterator)
            return tuple(tensor.numpy() for tensor in batch)
        except StopIteration:
            self.iterator = iter(self.dataset)
            batch = next(self.iterator)
            return tuple(tensor.numpy() for tensor in batch)

    def on_epoch_end(self):
        self.iterator = iter(self.dataset)
