import pyjevois
import libjevois as jevois
import cv2
import numpy as np
import os
import time
import cflib.crtp
import cflib.cpx
from cflib.cpx.transports import UARTTransport
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper
import serial
import traceback
import random
import struct
from collections import deque

class PythonSandbox:
    class Flow:
        def __init__(self, pos, flow_x, flow_y):
            if isinstance(pos, (tuple, list)) and len(pos) == 2:
                self.pos = pos
            else:
                raise ValueError("Invalid pos value")
            self.flow_x = flow_x
            self.flow_y = flow_y

    class Texton:
        def __init__(self):
            self.Y = None
            self.U = None
            self.V = None

    def __init__(self):
        self.is_pro = False
        self.log_level = 1
        self.smoothed_divergence = 0.0
        self.lp_factor = 0.05
        self.div_factor = -2
        self.real_obstacle_flag = 0
        self.use_harris = False
        self.old_gray = None
        self.p0 = None
        self.lk_params = {
            'winSize': (8, 8),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        }
        self.max_corners = 50
        self.quality_level = 0.1
        self.min_distance = 7
        self.block_size = 5
        self.TEXTONS_N_TEXTONS = 30
        self.TEXTONS_PATCH_SIZE = 16
        self.TEXTONS_DICTIONARY_PATH = "/jevois/data/shixu/3016.bin"
        self.TEXTONS_N_SAMPLES = 60
        self.TRAIN_FRAMES = 66
        self.dictionary = []
        self.distributions_batch = []
        self.all_distributions = []
        self.frame_number = 0
        self.initializing = True
        self.initial_training_frames_collected = 0
        self.background_model = None 
        self.start_delay = 15.0  
        self.start_time = time.time()
        self.started = False
        self.chi_square_history = deque(maxlen=100)
        self.threshold_k = 2
        self.consecutive_obstacle_frames = 0
        cflib.crtp.init_drivers(enable_serial_driver=True)
        self.packet = cflib.cpx.CPXPacket()
        self.packet.destination = cflib.cpx.CPXTarget.STM32
        self.packet.function = cflib.cpx.CPXFunction.APP
        self.frame = 0
        try:
            self.SerialSend = serial.Serial('/dev/ttyS0', 115200, timeout=2, write_timeout=2)
            jevois.LINFO('Serial port initialized successfully.')
        except Exception as e:
            jevois.LINFO(f'Failed to initialize serial port: {e}')
            self.SerialSend = None
        if os.path.isfile(self.TEXTONS_DICTIONARY_PATH):
            jevois.LINFO(f'Loading texton dictionary from: {self.TEXTONS_DICTIONARY_PATH}')
            if not self.load_texton_dictionary(self.TEXTONS_DICTIONARY_PATH, 
                                               self.TEXTONS_N_TEXTONS,
                                               self.TEXTONS_PATCH_SIZE):
                self.initializing = False
                jevois.LINFO("Failed to load texton dictionary.")
        else:
            self.initializing = False
            jevois.LINFO(f'Texton dictionary file not found: {self.TEXTONS_DICTIONARY_PATH}')
        self.precomputed_xs = None
        self.precomputed_ys = None
        self.last_width = None
        self.last_height = None
        self.output_dir = "/jevois/output"
        self.divergence_dir = os.path.join(self.output_dir, "divergence")
        self.obstacles_dir = os.path.join(self.output_dir, "obstacles")
        self.obstacles_texton_dir = os.path.join(self.obstacles_dir, "texton_distribution")
        self.obstacles_images_dir = os.path.join(self.obstacles_dir, "images")
        os.makedirs(self.divergence_dir, exist_ok=True)
        os.makedirs(self.obstacles_texton_dir, exist_ok=True)
        os.makedirs(self.obstacles_images_dir, exist_ok=True)
        self.divergence_csv_path = os.path.join(self.divergence_dir, "divergence.csv")
        try:
            self.divergence_file = open(self.divergence_csv_path, "w")
            self.divergence_file.write("frame_number,divergence\n")
            self.divergence_file.flush()
            jevois.LINFO(f"Divergence CSV initialized and opened in write mode at {self.divergence_csv_path}")
        except Exception as e:
            jevois.LINFO(f"Failed to initialize divergence CSV file: {e}")
            self.divergence_file = None
        self.obstacle_count = 0
        self.last_frame_time = None

    def __del__(self):
        try:
            if self.SerialSend and self.SerialSend.is_open:
                self.SerialSend.close()
                jevois.LINFO("Serial port closed.")
            if hasattr(self, 'divergence_file') and self.divergence_file:
                self.divergence_file.close()
                jevois.LINFO("Divergence CSV file closed.")
        except Exception as e:
            jevois.LINFO(f"Error during cleanup: {e}")
            traceback_str = traceback.format_exc()
            jevois.LINFO(f"Traceback: {traceback_str}")

    def set_log_level(self, level):
        self.log_level = level

    def log_info(self, message):
        if self.log_level >= 2:
            jevois.LINFO(message)

    def log_warning(self, message):
        if self.log_level >= 1:
            jevois.LINFO(f"[WARNING] {message}")

    def log_error(self, message):
        if self.log_level >= 0:
            jevois.LINFO(f"[ERROR] {message}")

    def start(self):
        pass

    def processNoUSB(self, inframe):
        try:
            current_time = time.time()
            if not self.started:
                if current_time - self.start_time >= self.start_delay:
                    self.started = True
                    jevois.LINFO("Startup delay completed. Starting processing.")
                else:
                    remaining_time = self.start_delay - (current_time - self.start_time)
                    jevois.LINFO(f"Waiting for startup delay: {remaining_time:.2f} seconds remaining.")
                    return
            inimg = inframe.getCvBGR()
            if inimg is None or inimg.size == 0:
                self.log_warning("Received empty frame.")
                return
            inimg = cv2.GaussianBlur(inimg, (5, 5), 0)
            inimg = cv2.medianBlur(inimg, 3)
            if not self.initializing and self.background_model is None:
                self.log_warning("Background model not initialized.")
                return
            if self.last_frame_time is None:
                self.last_frame_time = current_time
                return
            dt = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.process_frame_optical_flow(inimg, dt)
            self.process_frame_obstacle_detection(inimg)
            jevois.LINFO('Divergence is {:.3f}'.format(self.smoothed_divergence))
            jevois.LINFO('real_obstacle_flag={}'.format(self.real_obstacle_flag))
            x_float_scaled = self.smoothed_divergence * 1000.0
            x_int = int(round(x_float_scaled))
            if x_int < -32768:
                x_int = -32768
            elif x_int > 32767:
                x_int = 32767
            x_bytes = struct.pack('<h', x_int)
            x_byte1, x_byte2 = x_bytes
            y = self.real_obstacle_flag
            self.packet.data = [x_byte1, x_byte2, y]
            data_send = self.packet.wireData
            if len(data_send) > 100:
                raise Exception('Packet is too large!')
            buff = bytearray([0xFF, len(data_send)])
            buff.extend(data_send)
            checksum = 0
            for b in buff:
                checksum ^= b
            buff.append(checksum)
            if self.SerialSend and self.SerialSend.is_open:
                self.SerialSend.write(buff)
            else:
                self.log_warning("Serial port is not open. Cannot send data.")
            if self.divergence_file:
                self.divergence_file.write(f"{self.frame_number},{self.smoothed_divergence:.3f}\n")
                self.divergence_file.flush()
            if self.real_obstacle_flag == 1:
                self.obstacle_count += 1
                if hasattr(self, 'current_distribution') and self.current_distribution is not None:
                    texton_path = os.path.join(self.obstacles_texton_dir,
                                               f"frame_{self.frame_number}_texton.npy")
                    np.save(texton_path, self.current_distribution)
                    self.log_info(f"Saved texton distribution for frame {self.frame_number} at {texton_path}")
                else:
                    self.log_warning(f"No distribution data available for frame {self.frame_number}.")
                image_path = os.path.join(self.obstacles_images_dir, f"frame_{self.frame_number}.png")
                cv2.imwrite(image_path, inimg)
                self.log_info(f"Saved obstacle image for frame {self.frame_number} at {image_path}")
            time.sleep(0.001)
            self.frame_number += 1
        except Exception as e:
            self.log_error(f"Exception in processNoUSB: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")

    def process_frame_optical_flow(self, frame_bgr, dt):
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=self.use_harris
            )
            if self.p0 is not None:
                self.p0 = self.p0.reshape(-1, 1, 2)
            return
        if self.p0 is None or len(self.p0) == 0:
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=self.use_harris
            )
            if self.p0 is not None:
                self.p0 = self.p0.reshape(-1, 1, 2)
            self.old_gray = frame_gray.copy()
            return
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        if p1 is not None and st is not None:
            st = st.flatten()
            good_new = p1[st == 1].reshape(-1, 2)
            good_old = self.p0[st == 1].reshape(-1, 2)
            if len(good_new) < 2:
                self.log_warning("Not enough good features. Re-detecting features.")
                self.p0 = cv2.goodFeaturesToTrack(
                    frame_gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=self.block_size,
                    useHarrisDetector=self.use_harris
                )
                if self.p0 is not None:
                    self.p0 = self.p0.reshape(-1, 1, 2)
                self.old_gray = frame_gray.copy()
            else:
                vectors = [
                    self.Flow((gn[0], gn[1]), gn[0] - go[0], gn[1] - go[1])
                    for gn, go in zip(good_new, good_old)
                ]
                divergence = self.get_size_divergence(vectors, len(vectors), 40)
                self.smoothed_divergence = self.low_pass_filter_recursive(
                    divergence, self.smoothed_divergence,
                    self.lp_factor, self.div_factor, dt
                )
                self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.log_warning("Optical flow calculation failed. Re-detecting features.")
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=self.use_harris
            )
            if self.p0 is not None:
                self.p0 = self.p0.reshape(-1, 1, 2)
        self.old_gray = frame_gray.copy()

    def process_frame_obstacle_detection(self, frame_bgr):
        if len(self.dictionary) == 0:
            self.log_warning("Dictionary not loaded during obstacle detection.")
            self.real_obstacle_flag = 0
            self.current_distribution = None
            return
        frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
        Y = frame_yuv[:, :, 0].astype(np.float32)
        U = frame_yuv[:, :, 1].astype(np.float32)
        V = frame_yuv[:, :, 2].astype(np.float32)
        U_sub = cv2.resize(U, (U.shape[1] // 2, U.shape[0]), interpolation=cv2.INTER_AREA)
        V_sub = cv2.resize(V, (V.shape[1] // 2, V.shape[0]), interpolation=cv2.INTER_AREA)
        distribution = self.extract_texton_distribution_vectorized(
            Y, U_sub, V_sub, self.TEXTONS_PATCH_SIZE, self.TEXTONS_N_SAMPLES
        )
        self.current_distribution = distribution.copy()
        if self.initializing:
            self.distributions_batch.append(distribution)
            self.all_distributions.append(distribution)
            self.initial_training_frames_collected += 1
            self.real_obstacle_flag = 0
            if self.initial_training_frames_collected >= self.TRAIN_FRAMES:
                self.train_background_model()
                self.initializing = False
            return
        else:
            self.all_distributions.append(distribution)
            self.detect_obstacle(distribution)

    def train_background_model(self):
        if len(self.distributions_batch) == 0:
            self.log_error("No distributions available for background model training.")
            return False
        data = np.array(self.distributions_batch, dtype=np.float32)
        if data.size == 0:
            self.log_error("Background model training data is empty.")
            return False
        self.background_model = np.mean(data, axis=0)
        if np.sum(self.background_model) > 0:
            self.background_model /= np.sum(self.background_model)
        self.log_info("Background model trained successfully using Chi-Square method.")
        return True

    def detect_obstacle(self, distribution):
        if self.background_model is None:
            self.log_warning("Background model is not trained.")
            self.real_obstacle_flag = 0
            return
        if np.sum(distribution) > 0:
            distribution_normalized = distribution / np.sum(distribution)
        else:
            distribution_normalized = distribution
        chi_square_distance = 0.5 * np.sum(
            ((self.background_model - distribution_normalized) ** 2) /
            (self.background_model + distribution_normalized + 1e-10)
        )
        self.log_info(f"Chi-Square Distance: {chi_square_distance:.4f}")
        self.chi_square_history.append(chi_square_distance)
        if len(self.chi_square_history) >= 30:
            mean = np.mean(self.chi_square_history)
            std = np.std(self.chi_square_history)
            dynamic_threshold = mean + self.threshold_k * std
            self.log_info(f"Dynamic Threshold: {dynamic_threshold:.4f} (Mean: {mean:.4f}, Std: {std:.4f})")
        else:
            dynamic_threshold = 1
            self.log_info(f"Using fixed Threshold: {dynamic_threshold:.4f} (Insufficient history)")
        if chi_square_distance > dynamic_threshold:
            self.consecutive_obstacle_frames += 1
        else:
            self.consecutive_obstacle_frames = 0
        if self.consecutive_obstacle_frames >= 5:
            self.real_obstacle_flag = 1
        else:
            self.real_obstacle_flag = 0

    def load_texton_dictionary(self, dictionary_path, n_textons, patch_size):
        if not os.path.isfile(dictionary_path):
            self.log_error(f"Dictionary file does not exist: {dictionary_path}")
            return False
        try:
            with open(dictionary_path, 'rb') as f:
                header = f.read(8)
                if len(header) < 8:
                    self.log_error("Dictionary file header is incomplete.")
                    return False
                loaded_n_textons, loaded_patch_size = struct.unpack('<ii', header)
                if loaded_n_textons != n_textons or loaded_patch_size != patch_size:
                    self.log_error("Dictionary parameters mismatch.")
                    return False
                texton_vectors = []
                for _ in range(n_textons):
                    texton = self.Texton()
                    Y_bytes = f.read(4 * patch_size * patch_size)
                    if len(Y_bytes) < 4 * patch_size * patch_size:
                        self.log_error("Incomplete Y data in dictionary.")
                        return False
                    texton.Y = np.frombuffer(Y_bytes, dtype='<f4').reshape((patch_size, patch_size))
                    U_bytes = f.read(4 * (patch_size // 2) * patch_size)
                    if len(U_bytes) < 4 * (patch_size // 2) * patch_size:
                        self.log_error("Incomplete U data in dictionary.")
                        return False
                    texton.U = np.frombuffer(U_bytes, dtype='<f4').reshape(
                        (patch_size, patch_size // 2)
                    )
                    V_bytes = f.read(4 * (patch_size // 2) * patch_size)
                    if len(V_bytes) < 4 * (patch_size // 2) * patch_size:
                        self.log_error("Incomplete V data in dictionary.")
                        return False
                    texton.V = np.frombuffer(V_bytes, dtype='<f4').reshape(
                        (patch_size, patch_size // 2)
                    )
                    if texton.Y.size == 0 or texton.U.size == 0 or texton.V.size == 0:
                        self.log_error("Empty texton data encountered.")
                        return False
                    self.dictionary.append(texton)
                    texton_vectors.append(np.hstack((
                        texton.Y.flatten(),
                        texton.U.flatten(),
                        texton.V.flatten()
                    )))
                self.texton_matrix = np.array(texton_vectors, dtype=np.float32)
                self.texton_norm2 = (self.texton_matrix ** 2).sum(axis=1).reshape(1, -1)
                self.log_info("Texton dictionary loaded successfully.")
                return True
        except Exception as e:
            self.log_error(f"Error loading texton dictionary: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")
            return False

    def generate_random_samples(self, width, height, patch_size, n_samples):
        if self.last_width != width or self.last_height != height or self.precomputed_xs is None:
            rng = np.random.RandomState()
            max_x = width - patch_size
            max_y = height - patch_size
            if max_x <= 0 or max_y <= 0:
                self.precomputed_xs = np.array([], dtype=np.int32)
                self.precomputed_ys = np.array([], dtype=np.int32)
            else:
                xs = rng.randint(0, max_x, size=n_samples)
                ys = rng.randint(0, max_y, size=n_samples)
                valid_indices = (xs + patch_size <= width) & (ys + patch_size <= height)
                self.precomputed_xs = xs[valid_indices]
                self.precomputed_ys = ys[valid_indices]
            self.last_width = width
            self.last_height = height
        return self.precomputed_xs, self.precomputed_ys

    def extract_texton_distribution_vectorized(self, Y, U, V, patch_size, n_samples):
        n_textons = len(self.dictionary)
        distribution = np.zeros(n_textons, dtype=np.float32)
        if n_textons == 0:
            return distribution
        height, width = Y.shape
        xs, ys = self.generate_random_samples(width, height, patch_size, n_samples)
        if len(xs) == 0:
            return distribution
        patch_count = len(xs)
        xs_u = xs // 2
        patch_indices_y = ys[:, None] + np.arange(patch_size)
        patch_indices_x = xs[:, None] + np.arange(patch_size)
        patch_indices_x_u = xs_u[:, None] + np.arange(patch_size // 2)
        try:
            patch_Y = Y[patch_indices_y[:, :, None], patch_indices_x[:, None, :]]
            patch_U = U[patch_indices_y[:, :, None], patch_indices_x_u[:, None, :]]
            patch_V = V[patch_indices_y[:, :, None], patch_indices_x_u[:, None, :]]
        except IndexError as e:
            self.log_error(f"IndexError during patch extraction: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")
            return distribution
        patches_Y_flat = patch_Y.reshape(patch_count, -1)
        patches_U_flat = patch_U.reshape(patch_count, -1)
        patches_V_flat = patch_V.reshape(patch_count, -1)
        patches_flat = np.hstack((patches_Y_flat, patches_U_flat, patches_V_flat))
        patches_norm2 = np.sum(patches_flat ** 2, axis=1, keepdims=True)
        cross_term = np.dot(patches_flat, self.texton_matrix.T)
        distances_sq = patches_norm2 + self.texton_norm2 - 2 * cross_term
        distances_sq = np.maximum(distances_sq, 0.0)
        distances = np.sqrt(distances_sq, out=distances_sq)
        best_texton_indices = np.argmin(distances, axis=1)
        counts = np.bincount(best_texton_indices, minlength=n_textons)
        distribution[:len(counts)] = counts[:n_textons]
        if patch_count > 0:
            distribution /= patch_count
        return distribution

    def get_size_divergence(self, vectors, count, n_samples):
        if count < 2:
            return 0.0
        divs_sum = 0.0
        used_samples = 0
        max_samples = (count * (count - 1)) // 2
        if n_samples >= max_samples:
            n_samples = max_samples
        if n_samples == 0:
            for i in range(count):
                for j in range(i + 1, count):
                    dx1 = vectors[i].pos[0] - vectors[j].pos[0]
                    dy1 = vectors[i].pos[1] - vectors[j].pos[1]
                    distance_1_sq = dx1 * dx1 + dy1 * dy1
                    if distance_1_sq < 1e-10:
                        continue
                    dx2 = (vectors[i].pos[0] + vectors[i].flow_x) - (vectors[j].pos[0] + vectors[j].flow_x)
                    dy2 = (vectors[i].pos[1] + vectors[i].flow_y) - (vectors[j].pos[1] + vectors[j].flow_y)
                    distance_2_sq = dx2 * dx2 + dy2 * dy2
                    divs_sum += (np.sqrt(distance_2_sq) - np.sqrt(distance_1_sq)) / np.sqrt(distance_1_sq)
                    used_samples += 1
        else:
            rng = random.Random()
            indices = list(range(count))
            for _ in range(n_samples):
                i, j = rng.sample(indices, 2)
                dx1 = vectors[i].pos[0] - vectors[j].pos[0]
                dy1 = vectors[i].pos[1] - vectors[j].pos[1]
                distance_1_sq = dx1 * dx1 + dy1 * dy1
                if distance_1_sq < 1e-10:
                    continue
                dx2 = (vectors[i].pos[0] + vectors[i].flow_x) - (vectors[j].pos[0] + vectors[j].flow_x)
                dy2 = (vectors[i].pos[1] + vectors[i].flow_y) - (vectors[j].pos[1] + vectors[j].flow_y)
                distance_2_sq = dx2 * dx2 + dy2 * dy2
                divs_sum += (np.sqrt(distance_2_sq) - np.sqrt(distance_1_sq)) / np.sqrt(distance_1_sq)
                used_samples += 1
        if used_samples < 1:
            return 0.0
        return divs_sum / used_samples

    def low_pass_filter_recursive(self, new_divergence, previous_divergence, factor, div_factor, dt):
        if dt <= 0:
            self.log_warning("dt is non-positive in low_pass_filter_recursive.")
            return previous_divergence
        scaled_new = (new_divergence * div_factor) / dt
        return previous_divergence + (scaled_new - previous_divergence) * factor

    def main(self):
        pass
