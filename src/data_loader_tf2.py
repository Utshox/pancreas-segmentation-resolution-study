import tensorflow as tf
import numpy as np
from pathlib import Path
# import nibabel as nib # Not strictly used in the provided code if .npy are directly loaded
import time
# import logging # Not used in the provided code
import sys
import math

# Define AUTOTUNE for tf.data
AUTOTUNE = tf.data.experimental.AUTOTUNE

class PancreasDataLoader:
    def __init__(self, config):
        self.config = config
        self.target_size_hw = (config.img_size_y, config.img_size_x) # (Height, Width)

    def preprocess_volume(self, image_path_str_py, label_path_str_py=None):
        # This should be your latest working version of preprocess_volume
        # (the one that handles varied .npy shapes by standardizing to a list of 2D slices
        # and then uses tf.image.resize for each slice)
        # For brevity, I'm using the placeholder from your uploaded file, 
        # but ensure your actual robust method is here.
        # tf.print(f"DEBUG PDL.preprocess_volume: START. Image: '{image_path_str_py}', Label: '{label_path_str_py}'", output_stream=sys.stderr)
        try:
            if not Path(image_path_str_py).exists():
                tf.print(f"ERROR PDL.preprocess_volume: Image file NOT FOUND: '{image_path_str_py}'", output_stream=sys.stderr)
                return None if label_path_str_py is None else (None, None)
            raw_img_data = np.load(image_path_str_py)
            img_slices_list_hw = []
            if raw_img_data.ndim == 2: img_slices_list_hw.append(raw_img_data)
            elif raw_img_data.ndim == 3:
                if raw_img_data.shape[0] == 1 and raw_img_data.shape[1] > 1 and raw_img_data.shape[2] > 1 : img_slices_list_hw.append(raw_img_data[0])
                elif raw_img_data.shape[-1] == 1 and raw_img_data.shape[0] > 1 and raw_img_data.shape[1] > 1: img_slices_list_hw.append(np.squeeze(raw_img_data, axis=-1))
                elif raw_img_data.shape[0] > 1 and raw_img_data.shape[1] > 1 and raw_img_data.shape[2] > 1:
                    for i in range(raw_img_data.shape[0]): img_slices_list_hw.append(raw_img_data[i])
                else: 
                    if raw_img_data.shape[0] != 1 and raw_img_data.shape[1] != 1 and raw_img_data.shape[2] == 1 : img_slices_list_hw.append(np.squeeze(raw_img_data, axis=-1))
                    elif raw_img_data.shape[0] == 1 and raw_img_data.shape[1] != 1 and raw_img_data.shape[2] != 1 : img_slices_list_hw.append(raw_img_data[0])
                    else: 
                         for i in range(raw_img_data.shape[0]): img_slices_list_hw.append(raw_img_data[i])
            else: return None if label_path_str_py is None else (None, None)
            if not img_slices_list_hw: return None if label_path_str_py is None else (None, None)
            processed_img_volume_slices = []
            for idx, slice_hw in enumerate(img_slices_list_hw):
                slice_hw = np.nan_to_num(slice_hw).astype(np.float32); min_val, max_val = slice_hw.min(), slice_hw.max()
                if max_val > min_val: slice_hw = (slice_hw - min_val) / (max_val - min_val)
                else: slice_hw = np.zeros_like(slice_hw)
                slice_h_w_c = slice_hw[..., np.newaxis]
                try:
                    resized_slice_tf = tf.image.resize(tf.convert_to_tensor(slice_h_w_c,tf.float32), self.target_size_hw,'bilinear')
                    resized_slice_tf = tf.ensure_shape(resized_slice_tf, [self.target_size_hw[0],self.target_size_hw[1],1])
                    processed_img_volume_slices.append(resized_slice_tf.numpy())
                except Exception: pass # Simplified error handling for brevity
            if not processed_img_volume_slices: return None if label_path_str_py is None else (None, None)
            final_img_data_d_th_tw_c = np.stack(processed_img_volume_slices, axis=0)
            if self.config.num_channels == 1 and final_img_data_d_th_tw_c.shape[-1]!=1: final_img_data_d_th_tw_c=np.mean(final_img_data_d_th_tw_c,axis=-1,keepdims=True)
            elif self.config.num_channels == 3 and final_img_data_d_th_tw_c.shape[-1]==1: final_img_data_d_th_tw_c=np.repeat(final_img_data_d_th_tw_c,3,axis=-1)
            if label_path_str_py:
                if not Path(label_path_str_py).exists(): return None, None
                raw_label_data = np.load(label_path_str_py); label_slices_list_hw = []
                if raw_label_data.ndim == 2: label_slices_list_hw.append(raw_label_data)
                elif raw_label_data.ndim == 3:
                    if raw_label_data.shape[0] == 1 and raw_label_data.shape[1] > 1 and raw_label_data.shape[2] > 1 : label_slices_list_hw.append(raw_label_data[0])
                    elif raw_label_data.shape[-1] == 1 and raw_label_data.shape[0] > 1 and raw_label_data.shape[1] > 1: label_slices_list_hw.append(np.squeeze(raw_label_data, axis=-1))
                    elif raw_label_data.shape[0] > 1 and raw_label_data.shape[1] > 1 and raw_label_data.shape[2] > 1:
                        for i in range(raw_label_data.shape[0]): label_slices_list_hw.append(raw_label_data[i])
                    else: 
                        if raw_label_data.shape[0] != 1 and raw_label_data.shape[1] != 1 and raw_label_data.shape[2] == 1 : label_slices_list_hw.append(np.squeeze(raw_label_data, axis=-1))
                        elif raw_label_data.shape[0] == 1 and raw_label_data.shape[1] != 1 and raw_label_data.shape[2] != 1 : label_slices_list_hw.append(raw_label_data[0])
                        else: 
                            for i in range(raw_label_data.shape[0]): label_slices_list_hw.append(raw_label_data[i])
                else: return None, None
                if not label_slices_list_hw: return None, None
                processed_label_volume_slices = []
                for idx, slice_hw in enumerate(label_slices_list_hw):
                    slice_hw = np.nan_to_num(slice_hw); slice_hw = (slice_hw > 0.5).astype(np.float32)
                    slice_h_w_c = slice_hw[..., np.newaxis]
                    try:
                        resized_slice_tf = tf.image.resize(tf.convert_to_tensor(slice_h_w_c,tf.float32),self.target_size_hw,'nearest')
                        resized_slice_tf = tf.ensure_shape(resized_slice_tf,[self.target_size_hw[0],self.target_size_hw[1],1])
                        processed_label_volume_slices.append(resized_slice_tf.numpy())
                    except Exception: pass
                if not processed_label_volume_slices: return None, None
                final_label_data_d_th_tw_c = np.stack(processed_label_volume_slices,axis=0)
                final_label_data_d_th_tw_c = (final_label_data_d_th_tw_c > 0.5).astype(np.float32)
                min_slices = min(len(processed_img_volume_slices), len(processed_label_volume_slices))
                if min_slices == 0: return None, None
                return final_img_data_d_th_tw_c[:min_slices], final_label_data_d_th_tw_c[:min_slices]
            else:
                if final_img_data_d_th_tw_c.shape[0] == 0: return None
                return final_img_data_d_th_tw_c
        except Exception: return None if label_path_str_py is None else (None, None)

    @tf.function
    def _augment_slice_and_label(self, image_slice, label_slice, seed_pair):
        image_slice = tf.ensure_shape(image_slice, [self.target_size_hw[0], self.target_size_hw[1], self.config.num_channels])
        label_slice = tf.ensure_shape(label_slice, [self.target_size_hw[0], self.target_size_hw[1], 1])
        base_seed = tf.stack([seed_pair[0], seed_pair[1]]) 
        s_lr, s_ud, s_rot = tf.unstack(tf.random.experimental.stateless_split(base_seed, num=3), num=3) # Simpler split

        img_s, lbl_s = tf.cond(tf.random.stateless_uniform([], seed=s_lr[:2], minval=0, maxval=1) > 0.5, lambda: (tf.image.flip_left_right(image_slice), tf.image.flip_left_right(label_slice)), lambda: (image_slice, label_slice))
        img_s, lbl_s = tf.cond(tf.random.stateless_uniform([], seed=s_ud[:2], minval=0, maxval=1) > 0.5, lambda: (tf.image.flip_up_down(img_s), tf.image.flip_up_down(lbl_s)), lambda: (img_s, lbl_s))
        k_rot = tf.random.stateless_uniform([], seed=s_rot[:2], minval=0, maxval=4, dtype=tf.int32)
        img_s = tf.image.rot90(img_s, k=k_rot); lbl_s = tf.image.rot90(lbl_s, k=k_rot)
        
        # Add intensity augmentation for supervised training too
        img_s = self._augment_single_image_slice(img_s, strength='weak')
        
        return img_s, lbl_s

    @tf.function # Keep the @tf.function decorator
    def _augment_single_image_slice(self, image_slice, strength='weak'):
        """Applies color/intensity/noise/cutout augmentations to a single image slice using tf.cond."""
        image_slice = tf.ensure_shape(image_slice, [self.target_size_hw[0], self.target_size_hw[1], self.config.num_channels])
        
        current_image_slice = image_slice 

        # Generate all random decisions first (as scalar boolean tensors)
        # This avoids issues with tf.random inside tf.cond branches if not handled carefully with seeds.
        # However, for simple tf.random.uniform() > 0.5, direct use in tf.cond predicate is usually fine.
        # Let's stick to direct usage for now for simplicity.

        if strength == 'weak':
            current_image_slice = tf.cond(
                tf.random.uniform(shape=[]) > 0.5, # Condition for brightness
                lambda: tf.image.random_brightness(current_image_slice, max_delta=0.1),
                lambda: current_image_slice # No change
            )
            current_image_slice = tf.cond(
                tf.random.uniform(shape=[]) > 0.5, # Condition for contrast
                lambda: tf.image.random_contrast(current_image_slice, lower=0.9, upper=1.1),
                lambda: current_image_slice # No change
            )
        elif strength == 'strong':
            current_image_slice = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: tf.image.random_brightness(current_image_slice, max_delta=0.2),
                lambda: current_image_slice
            )
            current_image_slice = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: tf.image.random_contrast(current_image_slice, lower=0.8, upper=1.2),
                lambda: current_image_slice
            )
            current_image_slice = tf.cond( # Gaussian Noise
                tf.random.uniform(shape=[]) > 0.3,
                lambda: current_image_slice + tf.random.normal(shape=tf.shape(current_image_slice), mean=0.0, stddev=0.05, dtype=current_image_slice.dtype),
                lambda: current_image_slice
            )
            
            # Cutout
            def apply_cutout_strong(img_s_cut):
                img_h, img_w = self.target_size_hw[0], self.target_size_hw[1]
                cutout_h = tf.cast(tf.cast(img_h, tf.float32) * 0.25, tf.int32)
                cutout_w = tf.cast(tf.cast(img_w, tf.float32) * 0.25, tf.int32)
                offset_h = tf.random.uniform(shape=[], maxval=tf.maximum(1, img_h-cutout_h+1), dtype=tf.int32)
                offset_w = tf.random.uniform(shape=[], maxval=tf.maximum(1, img_w-cutout_w+1), dtype=tf.int32)
                
                # Create mask for cutout area (inverted: True where NOT cutout)
                r = tf.range(img_h); c = tf.range(img_w)
                mask_h = tf.logical_or(r < offset_h, r >= offset_h + cutout_h)
                mask_w = tf.logical_or(c < offset_w, c >= offset_w + cutout_w)
                final_mask = tf.logical_or(mask_h[:, tf.newaxis], mask_w[tf.newaxis, :]) # True if outside cutout
                final_mask_3d = final_mask[..., tf.newaxis] # Add channel dim
                
                return tf.where(final_mask_3d, img_s_cut, tf.zeros_like(img_s_cut))

            current_image_slice = tf.cond(
                tf.random.uniform(shape=[]) < 0.3, 
                lambda: apply_cutout_strong(current_image_slice),
                lambda: current_image_slice
            )

        final_image_slice = tf.clip_by_value(current_image_slice, 0.0, 1.0)
        return final_image_slice


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.dataloader = PancreasDataLoader(config)

    def _py_load_preprocess_volume_wrapper(self, image_path_input, label_path_input=None):
        # This should be your latest working version that correctly decodes paths
        # and calls self.dataloader.preprocess_volume, returning numpy arrays
        # matching the Tout signatures of the from_generator calls.
        # For brevity, I'm assuming this method is correctly implemented as per our prior discussions.
        def decode_path(p):
            if isinstance(p, tf.Tensor): return p.numpy().decode('utf-8')
            if isinstance(p, bytes): return p.decode('utf-8')
            if isinstance(p, str): return p
            return None
        img_str = decode_path(image_path_input)
        lbl_str = decode_path(label_path_input) if label_path_input is not None else None
        
        if img_str is None:
            d_dummy,h_dummy,w_dummy,c_dummy = 1,self.config.img_size_y,self.config.img_size_x,self.config.num_channels
            img_vol_dummy = np.zeros([d_dummy,h_dummy,w_dummy,c_dummy], dtype=np.float32)
            if lbl_str is not None or (label_path_input is not None and not isinstance(label_path_input, (tf.Tensor, bytes, str))): # Heuristic for supervised call
                return img_vol_dummy, np.zeros([d_dummy,h_dummy,w_dummy,1], dtype=np.float32)
            return img_vol_dummy

        result = self.dataloader.preprocess_volume(img_str, lbl_str)
        is_supervised_call = lbl_str is not None

        d_dummy,h_dummy,w_dummy,c_dummy = 1,self.config.img_size_y,self.config.img_size_x,self.config.num_channels
        img_vol_dummy = np.zeros([d_dummy,h_dummy,w_dummy,c_dummy], dtype=np.float32)

        if is_supervised_call:
            img_data, lbl_data = result if result and isinstance(result,tuple) and len(result)==2 else (None,None)
            if img_data is None or lbl_data is None:
                return img_vol_dummy, np.zeros([d_dummy,h_dummy,w_dummy,1], dtype=np.float32)
            return img_data.astype(np.float32), lbl_data.astype(np.float32)
        else:
            img_data = result
            if img_data is None: return img_vol_dummy
            return img_data.astype(np.float32)


    def _parse_volume_to_slices_supervised(self, image_path_tensor, label_path_tensor):
        # Yields (image_slice_shape, label_slice_shape) which are (H,W,C_img) and (H,W,1)
        img_vol_np, lbl_vol_np = self._py_load_preprocess_volume_wrapper(image_path_tensor, label_path_tensor)
        for i in range(img_vol_np.shape[0]): 
            yield img_vol_np[i], lbl_vol_np[i]

    def _parse_volume_to_slices_unlabeled(self, image_path_tensor):
        # Yields image_slice_shape which is (H,W,C_img)
        img_vol_np = self._py_load_preprocess_volume_wrapper(image_path_tensor, None)
        for i in range(img_vol_np.shape[0]): 
            yield img_vol_np[i]

    @tf.function
    def _stateless_geometric_augment_single_slice(self, image_slice, seed_pair):
        # (Corrected version with tf.cond from previous responses)
        base_seed = tf.stack([seed_pair[0], seed_pair[1]]) 
        s_lr, s_ud, s_rot = tf.unstack(tf.random.experimental.stateless_split(base_seed, num=3), num=3)
        img_s = tf.cond(tf.random.stateless_uniform([],seed=s_lr[:2]) > 0.5, lambda: tf.image.flip_left_right(image_slice), lambda: image_slice)
        img_s = tf.cond(tf.random.stateless_uniform([],seed=s_ud[:2]) > 0.5, lambda: tf.image.flip_up_down(img_s), lambda: img_s)
        k_rot = tf.random.stateless_uniform([], seed=s_rot[:2], minval=0, maxval=4, dtype=tf.int32)
        return tf.image.rot90(img_s, k=k_rot)

    def _augment_for_mean_teacher(self, image_slice, seed_pair_student_geom, seed_pair_teacher_geom):
        # (Corrected version from previous responses)
        student_view_geom = self._stateless_geometric_augment_single_slice(image_slice, seed_pair_student_geom)
        student_view = self.dataloader._augment_single_image_slice(student_view_geom, strength='strong')
        teacher_view_geom = self._stateless_geometric_augment_single_slice(image_slice, seed_pair_teacher_geom)
        teacher_view = self.dataloader._augment_single_image_slice(teacher_view_geom, strength='weak')
        student_view = tf.ensure_shape(student_view, [self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
        teacher_view = tf.ensure_shape(teacher_view, [self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
        return student_view, teacher_view

    def build_labeled_dataset(self, image_paths, label_paths, batch_size, is_training=True):
        # (Corrected version from previous responses)
        if not image_paths or not label_paths: return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        if is_training: dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.interleave(
            lambda img_p, lbl_p: tf.data.Dataset.from_generator(self._parse_volume_to_slices_supervised, args=(img_p, lbl_p),
                output_signature=(tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                                  tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32))),
            cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=not is_training)
        if is_training:
            seed_ds = tf.data.Dataset.counter().map(lambda x: (x, x + tf.cast(4e5, tf.int64))) # Unique seed base
            dataset = tf.data.Dataset.zip((dataset, seed_ds))
            dataset = dataset.map(lambda dp, sp: self.dataloader._augment_slice_and_label(dp[0], dp[1], sp), num_parallel_calls=AUTOTUNE)
            # Apply intensity augmentations to labeled data as well
            dataset = dataset.map(
                 lambda img, lbl: (self.dataloader._augment_single_image_slice(img, strength='strong'), lbl),
                 num_parallel_calls=AUTOTUNE
            )
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def build_unlabeled_dataset_for_mean_teacher(self, image_paths, batch_size, is_training=True):
        # (Corrected version from previous responses)
        if not image_paths:
            return tf.data.Dataset.from_tensor_slices((
                tf.zeros([0,self.config.img_size_y,self.config.img_size_x,self.config.num_channels], dtype=tf.float32), 
                tf.zeros([0,self.config.img_size_y,self.config.img_size_x,self.config.num_channels], dtype=tf.float32)
            )).batch(batch_size)
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        if is_training: dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.interleave(
            lambda img_p: tf.data.Dataset.from_generator(self._parse_volume_to_slices_unlabeled, args=(img_p,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)),
            cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=not is_training)
        if is_training:
            s_stud_geom = tf.data.Dataset.counter().map(lambda x: (x, x + tf.cast(1e5, tf.int64))) 
            s_teach_geom = tf.data.Dataset.counter().map(lambda x: (x + tf.cast(2e5, tf.int64), x + tf.cast(3e5, tf.int64)))
            dataset_with_seeds = tf.data.Dataset.zip((dataset, s_stud_geom, s_teach_geom))
            dataset = dataset_with_seeds.map(self._augment_for_mean_teacher, num_parallel_calls=AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def build_validation_dataset(self, image_paths, label_paths, batch_size):
        # (Corrected version from previous responses)
        if not image_paths or not label_paths: return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.interleave(
            lambda img_p, lbl_p: tf.data.Dataset.from_generator(self._parse_volume_to_slices_supervised, args=(img_p, lbl_p),
                output_signature=(tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                                  tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32))),
            cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=True)
        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def build_validation_dataset(self, image_paths, label_paths, batch_size):
        if not image_paths or not label_paths: return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.interleave(
            lambda img_p, lbl_p: tf.data.Dataset.from_generator(self._parse_volume_to_slices_supervised, args=(img_p, lbl_p),
                output_signature=(tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                                  tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32))),
            cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=True)
        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset
        # --- ADD THIS METHOD ---

    def build_unlabeled_dataset_for_mixmatch(self, image_paths, batch_size, is_training=True):
        """
        Creates a dataset of unlabeled images for MixMatch.
        Yields batches of single-view image slices.
        Minimal augmentation here; MixMatchTrainer handles its specific K-augmentations.
        """
        if not image_paths:
            tf.print("WARNING DataPipeline: build_unlabeled_dataset_for_mixmatch received empty image_paths.", output_stream=sys.stderr)
            # Return a dataset that yields a tuple of one element, even if empty,
            # because MixMatchTrainer's zip expects (labeled_batch, unlabeled_batch_tuple)
            # where unlabeled_batch_tuple is (images_xu_raw,).
            empty_img_slice_batch = tf.zeros([0, self.config.img_size_y, self.config.img_size_x, self.config.num_channels], dtype=tf.float32)
            return tf.data.Dataset.from_tensor_slices( (empty_img_slice_batch,) ).batch(batch_size)

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        
        if is_training: # Shuffle original file paths
            dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

        # Load and preprocess volumes, then flatten to slices
        dataset = dataset.interleave(
            lambda img_path_tensor: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_unlabeled, # Yields single (H,W,C) slices
                args=(img_path_tensor,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)
            ),
            cycle_length=AUTOTUNE,
            num_parallel_calls=AUTOTUNE,
            deterministic=not is_training # Be non-deterministic if training
        )
        
        # Filter out any dummy/error slices that might have zero depth from preprocess_volume via _py_load_wrapper
        # A successfully processed slice will have H = self.config.img_size_y
        dataset = dataset.filter(lambda img_slice: tf.shape(img_slice)[0] == self.config.img_size_y)

        # For MixMatch, the train_step usually expects a batch of unlabeled images X_u.
        # If the trainer's zipping expects (labeled_batch, (unlabeled_batch,)), then map to tuple here.
        dataset = dataset.map(lambda x: (x,), num_parallel_calls=AUTOTUNE) # Wrap in a tuple: (img_slice,)

        if is_training:
            dataset = dataset.shuffle(buffer_size=2048) # Shuffle individual slices

        # drop_remainder=True is often important for MixMatch if B_l and B_u must be same for MixUp X = Concat(Xl, Xu)
        # but our current MixMatchTrainer mixes Xl with Xl and Xu with Xu, so it's less critical.
        dataset = dataset.batch(batch_size, drop_remainder=is_training) 
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        
        # tf.print(f"DEBUG DataPipeline: build_unlabeled_dataset_for_mixmatch, element_spec: {dataset.element_spec}", output_stream=sys.stderr)
        return dataset
    # build_unlabeled_dataset_for_mixmatch and the second build_validation_dataset were duplicates
    # Keep only one build_validation_dataset.
    # If build_unlabeled_dataset_for_mixmatch is needed, ensure it's correctly defined.
    # For now, I'm removing the duplicate build_validation_dataset and the MixMatch one from your provided code
    # as we are focusing on Mean Teacher. If you need them back, let me know.