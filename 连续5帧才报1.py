import pyjevois
import libjevois as jevois
import cv2
import numpy as np
import os
import time
import logging
import cflib.crtp
import cflib.cpx
from cflib.cpx.transports import UARTTransport
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper
import serial
import struct
import traceback
import random

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
        self.log_level = 2
        self.log_info("Initializing PythonTest...")
        self.timer = jevois.Timer("test", 100, jevois.LOG_INFO)
        self.old_gray = None
        self.p0 = None
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        }
        self.max_corners = 50
        self.quality_level = 0.15
        self.min_distance = 5
        self.block_size = 5
        self.use_harris = False
        self.k = 0.04
        self.div_factor = -2.25
        self.lp_factor = 0.1
        self.smoothed_divergence = 0.0
        self.divergence_history = []
        self.prev_time = time.time()
        self.scale_factor = 1
        self.TEXTONS_N_TEXTONS = 20
        self.TEXTONS_PATCH_SIZE = 8
        self.TEXTONS_DICTIONARY_PATH = "/jevois/data/shixu/2008.bin"
        self.TEXTONS_N_SAMPLES = 100
        self.TRAIN_FRAMES = 50
        self.UPDATE_INTERVAL = 10
        self.KMEANS_N_CLUSTERS = 5
        self.KMEANS_MAX_ITER = 100
        self.KMEANS_EPSILON = 0.1
        self.KMEANS_ATTEMPTS = 5
        self.KMEANS_FLAGS = cv2.KMEANS_PP_CENTERS
        self.OBSTACLE_LABEL = "Obstacle Detected!"
        self.NO_OBSTACLE_LABEL = "No Obstacle."
        self.INITIALIZING_LABEL = "Initializing..."
        self.BACKGROUND_DISTANCE_THRESHOLD = 0.3
        self.DISTANCE_THRESHOLD = 0.5
        self.MERGE_DISTANCE_THRESHOLD = 0.3
        self.MAX_OBSTACLE_CENTERS = 10
        self.dictionary = []
        self.distributions_batch = []
        self.update_distributions_batch = []
        self.all_distributions = []
        self.cluster_centers_history = []
        self.initial_cluster_centers = None
        self.cluster_labels = []
        self.obstacle_centers = []
        self.frame_number = 0
        self.obstacle_detected = False
        self.initializing = True
        self.initial_training_frames_collected = 0
        self.current_obstacle_flag = 0
        self.consecutive_obstacle_frames = 0
        cflib.crtp.init_drivers(enable_serial_driver=True)
        self.packet = cflib.cpx.CPXPacket()
        self.packet.destination = cflib.cpx.CPXTarget.STM32
        self.packet.function = cflib.cpx.CPXFunction.APP
        self.frame = 0
        self.SerialSend = serial.Serial('/dev/ttyS0', 115200, timeout=2, write_timeout=2)
        if os.path.isfile(self.TEXTONS_DICTIONARY_PATH):
            self.log_info(f"Loading texton dictionary from: {self.TEXTONS_DICTIONARY_PATH}")
            if not self.load_texton_dictionary(self.TEXTONS_DICTIONARY_PATH, self.TEXTONS_N_TEXTONS, self.TEXTONS_PATCH_SIZE):
                self.initializing = False
                self.log_error("Failed to load texton dictionary.")
        else:
            self.initializing = False
            self.log_error(f"Texton dictionary file not found: {self.TEXTONS_DICTIONARY_PATH}")
        self.precomputed_xs = None
        self.precomputed_ys = None
        self.last_width = None
        self.last_height = None
        self.log_info("PythonTest initialized successfully.")
    def __del__(self):
        try:
            self.SerialSend.close()
            self.log_info("Serial port closed.")
        except Exception as e:
            self.log_error(f"Error during cleanup: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")
    def set_log_level(self, level):
        self.log_level = level
    def log_info(self, message):
        if self.log_level >= 2:
            if self.is_pro:
                jevois.sendLog(jevois.LOG_INFO, message)
            else:
                if hasattr(jevois, 'info'):
                    jevois.info(message)
                else:
                    print(f"[INFO] {message}")
    def log_warning(self, message):
        if self.log_level >= 1:
            if self.is_pro:
                jevois.sendLog(jevois.LOG_WARNING, message)
            else:
                if hasattr(jevois, 'warning'):
                    jevois.warning(message)
                else:
                    print(f"[WARNING] {message}")
    def log_error(self, message):
        if self.log_level >= 0:
            if self.is_pro:
                jevois.sendLog(jevois.LOG_ERROR, message)
            else:
                if hasattr(jevois, 'error'):
                    jevois.error(message)
                elif hasattr(jevois, 'warning'):
                    jevois.warning(message)
                elif hasattr(jevois, 'info'):
                    jevois.info(message)
                else:
                    print(f"[ERROR] {message}")
    def start(self):
        self.log_info("Start method called.")
        pass
    def process(self, inframe, outframe):
        self.SerialSend.reset_output_buffer()
        try:
            inimg = inframe.getCvBGR()
            if inimg is None or inimg.size == 0:
                self.log_warning("Received empty frame.")
                return
            inimg = cv2.resize(inimg, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            if not self.initializing and len(self.dictionary) == 0:
                self.log_warning("Dictionary not loaded.")
                cv2.putText(inimg, "Dictionary not loaded", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                outframe.sendCv(inimg)
                return
            self.timer.start()
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            self.log_info(f"Processing frame number: {self.frame_number + 1}, dt: {dt:.4f}s")
            frame_bgr = self.process_frame_optical_flow(inimg, dt)
            frame_bgr = self.process_frame_obstacle_detection(frame_bgr)
            outframe.sendCv(frame_bgr)
            jevois.LINFO('Divergence is {:.2f}'.format(self.smoothed_divergence))
            jevois.LINFO('Obstacle Flag={}'.format(self.current_obstacle_flag))
            self.log_info('Controlled divergence is {:.2f}'.format(self.smoothed_divergence))
            self.log_info('Parameters are 0 and 1')
            self.log_info('Controlled divergence is {:.2f}'.format(self.smoothed_divergence))
            self.log_info('Parameters are 0 and 1')
            x = int(self.smoothed_divergence * 100)
            y = self.current_obstacle_flag
            data = struct.pack('>hb', x, y)
            buff = bytearray([0xFF, len(data)])
            buff.extend(data)
            checksum = 0
            for byte in buff:
                checksum ^= byte
            buff.append(checksum)
            self.log_info(f"Sending buff: {buff.hex()} (y={y}, x={x})")
            jevois.LINFO('buff is {}'.format(buff))
            self.SerialSend.write(buff)
            time.sleep(0.001)
            self.frame_number += 1
        except Exception as e:
            self.log_error(f"Exception in process: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")
    def process_frame_optical_flow(self, frame_bgr, dt):
        self.log_info("Starting optical flow processing.")
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.old_gray is None:
            self.log_info("Old gray frame is None. Initializing optical flow.")
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=self.use_harris,
                k=self.k
            )
            if self.p0 is not None:
                self.p0 = self.p0.reshape(-1, 1, 2)
                self.log_info(f"Detected {len(self.p0)} good features to track.")
            return frame_bgr
        if self.p0 is None or len(self.p0) == 0:
            self.log_info("No good features to track. Re-detecting features.")
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=self.use_harris,
                k=self.k
            )
            if self.p0 is not None:
                self.p0 = self.p0.reshape(-1, 1, 2)
                self.log_info(f"Re-detected {len(self.p0)} good features.")
            self.old_gray = frame_gray.copy()
            cv2.putText(frame_bgr, "Re-detecting features", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame_bgr
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        if p1 is not None and st is not None:
            st = st.flatten()
            good_new = p1[st == 1].reshape(-1, 2)
            good_old = self.p0[st == 1].reshape(-1, 2)
            self.log_info(f"Found {len(good_new)} good features after optical flow.")
            if len(good_new) < 2:
                self.log_warning("Not enough good features. Re-detecting features.")
                self.p0 = cv2.goodFeaturesToTrack(
                    frame_gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=self.block_size,
                    useHarrisDetector=self.use_harris,
                    k=self.k
                )
                if self.p0 is not None:
                    self.p0 = self.p0.reshape(-1, 1, 2)
                    self.log_info(f"Re-detected {len(self.p0)} good features.")
                self.old_gray = frame_gray.copy()
                cv2.putText(frame_bgr, "Re-detecting features", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                vectors = [
                    self.Flow((gn[0], gn[1]), gn[0] - go[0], gn[1] - go[1])
                    for gn, go in zip(good_new, good_old)
                ]
                divergence = self.get_size_divergence(vectors, len(vectors), 20)
                self.smoothed_divergence = self.low_pass_filter_recursive(divergence, self.smoothed_divergence, self.lp_factor, self.div_factor, dt)
                divergence_text = f"Divergence: {self.smoothed_divergence:.2f}" if not np.isnan(self.smoothed_divergence) else "Divergence: N/A"
                cv2.putText(frame_bgr, divergence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for flow in vectors:
                    a, b = flow.pos
                    cv2.circle(frame_bgr, (int(a), int(b)), 2, (0, 255, 0), -1)
                self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.log_warning("Optical flow calculation failed. Re-detecting features.")
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=self.use_harris,
                k=self.k
            )
            if self.p0 is not None:
                self.p0 = self.p0.reshape(-1, 1, 2)
                self.log_info(f"Re-detected {len(self.p0)} good features.")
            self.old_gray = frame_gray.copy()
            cv2.putText(frame_bgr, "Re-detecting features", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self.old_gray = frame_gray.copy()
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_text = f"FPS: {fps:.2f}" if not np.isnan(fps) else "FPS: N/A"
        outheight = frame_bgr.shape[0]
        cv2.putText(frame_bgr, fps_text, (10, outheight - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.log_info(f"Optical flow processing completed. FPS: {fps:.2f}")
        return frame_bgr
    def process_frame_obstacle_detection(self, frame_bgr):
        self.log_info("Starting obstacle detection.")
        if len(self.dictionary) == 0:
            self.log_warning("Dictionary not loaded during obstacle detection.")
            cv2.putText(frame_bgr, "Dictionary not loaded", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.current_obstacle_flag = 0
            return frame_bgr
        frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
        Y = frame_yuv[:, :, 0].astype(np.float32)
        U = frame_yuv[:, :, 1].astype(np.float32)
        V = frame_yuv[:, :, 2].astype(np.float32)
        U_sub = cv2.resize(U, (U.shape[1] // 2, U.shape[0]), interpolation=cv2.INTER_AREA)
        V_sub = cv2.resize(V, (V.shape[1] // 2, V.shape[0]), interpolation=cv2.INTER_AREA)
        distribution = self.extract_texton_distribution_vectorized(Y, U_sub, V_sub, self.TEXTONS_PATCH_SIZE, self.TEXTONS_N_SAMPLES)
        if self.initializing:
            self.log_info("Initializing: Collecting distributions.")
            self.distributions_batch.append(distribution)
            self.all_distributions.append(distribution)
            self.initial_training_frames_collected += 1
            cv2.putText(frame_bgr, self.INITIALIZING_LABEL, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            self.current_obstacle_flag = 0
            if self.initial_training_frames_collected >= self.TRAIN_FRAMES:
                self.log_info("Reached TRAIN_FRAMES. Perform initial KMeans training.")
                if self.train_initial_kmeans():
                    self.initializing = False
                    self.log_info("Initialization complete using K-Means.")
            return frame_bgr
        else:
            self.all_distributions.append(distribution)
            self.update_distributions_batch.append(distribution)
            if len(self.update_distributions_batch) == self.UPDATE_INTERVAL:
                self.log_info("Performing KMeans update with recent distributions.")
                self.update_kmeans()
                self.update_distributions_batch = []
            is_obstacle, label, color = self.detect_obstacle(distribution)
            if is_obstacle:
                self.consecutive_obstacle_frames += 1
            else:
                self.consecutive_obstacle_frames = 0
            if self.consecutive_obstacle_frames >= 5:
                self.current_obstacle_flag = 1
            else:
                self.current_obstacle_flag = 0
            cv2.putText(frame_bgr, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            return frame_bgr
    def train_initial_kmeans(self):
        if len(self.distributions_batch) == 0:
            self.log_error("No distributions available for initial KMeans training.")
            return False
        data = np.array(self.distributions_batch, dtype=np.float32)
        if data.size == 0:
            self.log_error("Initial KMeans training data is empty.")
            return False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.KMEANS_MAX_ITER, self.KMEANS_EPSILON)
        try:
            _, labels, centers = cv2.kmeans(
                data, self.KMEANS_N_CLUSTERS, None,
                criteria,
                self.KMEANS_ATTEMPTS,
                self.KMEANS_FLAGS
            )
            self.cluster_centers_history.append(centers)
            self.initial_cluster_centers = centers.copy()
            self.cluster_labels = ['Normal'] * self.KMEANS_N_CLUSTERS
            self.log_info("Initial KMeans training completed successfully.")
            return True
        except Exception as e:
            self.log_error(f"Error during initial KMeans training: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")
            return False
    def update_kmeans(self):
        if len(self.update_distributions_batch) < self.UPDATE_INTERVAL:
            self.log_warning("Not enough distributions for KMeans update.")
            return []
        data = np.array(self.update_distributions_batch, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.KMEANS_MAX_ITER, self.KMEANS_EPSILON)
        try:
            _, labels, centers = cv2.kmeans(
                data, self.KMEANS_N_CLUSTERS, None,
                criteria,
                self.KMEANS_ATTEMPTS,
                self.KMEANS_FLAGS
            )
            self.cluster_centers_history.append(centers)
            new_obstacle_centers = []
            for idx, c_new in enumerate(centers):
                distances_to_initial = np.linalg.norm(self.initial_cluster_centers - c_new, axis=1)
                min_distance_to_initial = np.min(distances_to_initial)
                if min_distance_to_initial > self.BACKGROUND_DISTANCE_THRESHOLD:
                    is_new_obstacle = True
                    for existing_center in self.obstacle_centers:
                        distance_to_existing = np.linalg.norm(existing_center - c_new)
                        if distance_to_existing < self.MERGE_DISTANCE_THRESHOLD:
                            new_center = (existing_center + c_new) / 2
                            self.obstacle_centers.remove(existing_center)
                            self.obstacle_centers.append(new_center)
                            is_new_obstacle = False
                            break
                    if is_new_obstacle:
                        if len(self.obstacle_centers) >= self.MAX_OBSTACLE_CENTERS:
                            self.obstacle_centers.pop(0)
                        self.obstacle_centers.append(c_new)
                        new_obstacle_centers.append(c_new)
            self.log_info(f"Updated obstacle centers: {new_obstacle_centers}")
            return new_obstacle_centers
        except Exception as e:
            self.log_error(f"Error in update_kmeans: {e}")
            traceback_str = traceback.format_exc()
            self.log_error(f"Traceback: {traceback_str}")
            return []
    def detect_obstacle(self, distribution):
        is_obstacle = False
        if self.initial_cluster_centers is not None and len(self.initial_cluster_centers) > 0:
            distances_to_initial = np.linalg.norm(self.initial_cluster_centers - distribution, axis=1)
            min_distance_to_initial = np.min(distances_to_initial)
            if min_distance_to_initial <= self.BACKGROUND_DISTANCE_THRESHOLD:
                return False, self.NO_OBSTACLE_LABEL, (0, 255, 0)
        if len(self.obstacle_centers) > 0:
            distances = np.linalg.norm(self.obstacle_centers - distribution, axis=1)
            if np.any(distances < self.DISTANCE_THRESHOLD):
                return True, self.OBSTACLE_LABEL, (0, 0, 255)
            else:
                return False, self.NO_OBSTACLE_LABEL, (0, 255, 0)
        else:
            return False, self.NO_OBSTACLE_LABEL, (0, 255, 0)
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
                    texton.U = np.frombuffer(U_bytes, dtype='<f4').reshape((patch_size, patch_size // 2))
                    V_bytes = f.read(4 * (patch_size // 2) * patch_size)
                    if len(V_bytes) < 4 * (patch_size // 2) * patch_size:
                        self.log_error("Incomplete V data in dictionary.")
                        return False
                    texton.V = np.frombuffer(V_bytes, dtype='<f4').reshape((patch_size, patch_size // 2))
                    if texton.Y.size == 0 or texton.U.size == 0 or texton.V.size == 0:
                        self.log_error("Empty texton data encountered.")
                        return False
                    self.dictionary.append(texton)
                    texton_vectors.append(np.hstack((texton.Y.flatten(), texton.U.flatten(), texton.V.flatten())))
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
            rng = np.random.RandomState(42)
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
            rng = random.Random(42)
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
        self.log_info("PythonTest main method started.")
        pass
if __name__ == "__main__":
    sandbox = PythonSandbox()
    sandbox.set_log_level(2)
    sandbox.main()
