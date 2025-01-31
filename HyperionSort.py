import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math
import pandas as pd
import psutil
import time
import warnings
from enum import Enum
import cProfile
import io
import pstats
from collections import deque
import heapq
from functools import partial, lru_cache
import logging
import sys
from contextlib import contextmanager
import pickle
import os
from datetime import datetime
import threading
import multiprocessing as mp
import random
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from scipy.stats import uniform, randint
from scipy.stats import skew, kurtosis, entropy
from lz4.frame import compress, decompress
import tensorflow as tf
import asyncio
import numba
import json
import gc
from multiprocessing import shared_memory
import mmap
import socket
import dask.array as da
import ray
from tqdm import tqdm
import xgboost as xgb
from typing import Dict, Any, List, Union, Optional, Tuple, Callable, Generator
import lightgbm as lgb
from catboost import CatBoostClassifier
from retrying import retry
import pytest
from hilbertcurve.hilbertcurve import HilbertCurve
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category20
from bokeh.io import output_notebook
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(processName)s - %(threadName)s'
)
logger = logging.getLogger(__name__)

def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_io = psutil.disk_io_counters()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(f"Memory Usage: {memory_info.percent}%")
    logger.info(f"Disk Read: {disk_io.read_bytes / (1024 * 1024):.2f} MB")
    logger.info(f"Disk Write: {disk_io.write_bytes / (1024 * 1024):.2f} MB")

def plot_performance_metrics(results):
    df = pd.DataFrame(results)
    
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['size', 'time'])
    
    if len(df) == 0:
        print("Không có dữ liệu hợp lệ để vẽ biểu đồ!")
        return
        
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='size', y='time', hue='strategy', marker='o', ci=None)
    plt.title('Execution Time vs Data Size')
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend(title='Strategy')
    plt.grid(True)
    plt.show()

def plot_interactive_performance_metrics(results):
    output_notebook()
    df = pd.DataFrame(results)
    source = ColumnDataSource(df)
    
    p = figure(title="Execution Time vs Data Size", x_axis_label='Data Size', y_axis_label='Execution Time (seconds)', plot_width=800, plot_height=400)
    
    strategies = df['strategy'].unique()
    colors = Category20[len(strategies)]
    
    for i, strategy in enumerate(strategies):
        strategy_df = df[df['strategy'] == strategy]
        p.line(strategy_df['size'], strategy_df['time'], legend_label=strategy, line_width=2, color=colors[i])
        p.circle(strategy_df['size'], strategy_df['time'], fill_color=colors[i], size=8)
    
    p.legend.title = 'Strategy'
    show(p)

def analyze_performance_metrics(results):
    df = pd.DataFrame(results)
    df['log_size'] = np.log(df['size'])
    df['log_time'] = np.log(df['time'])
    
    model = sm.OLS(df['log_time'], sm.add_constant(df[['log_size', 'strategy']]))
    results = model.fit()
    print(results.summary())
    
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def critical_function():
    log_system_metrics()
    sorter = EnhancedHyperionSort()
    data = np.random.randint(0, 1000, size=1_000_000)
    sorted_data, stats = asyncio.run(sorter.sort(data))
    logger.info(f"Sorting completed. Execution time: {stats.execution_time:.4f} seconds")

class WrappedXGBClassifier(xgb.XGBClassifier, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator_type = "classifier"

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        return self

    def __sklearn_tags__(self):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'allow_nan': False,
            'stateless': False,
            'binary_only': False,
            'requires_fit': True,
            'requires_y': True,
            'pairwise': False,
            'preserves_dtype': [np.float64, np.float32]
        }

class WrappedLGBMClassifier(lgb.LGBMClassifier, BaseEstimator):
    def __init__(self, random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self._estimator_type = "classifier"

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        return self

    def __sklearn_tags__(self):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'allow_nan': False,
            'stateless': False,
            'binary_only': False,
            'requires_fit': True,
            'requires_y': True,
            'pairwise': False,
            'preserves_dtype': [np.float64, np.float32]
        }

class WrappedCatBoostClassifier(CatBoostClassifier, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator_type = "classifier"

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        return self

    def __sklearn_tags__(self):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'allow_nan': False,
            'stateless': False,
            'binary_only': False,
            'requires_fit': True,
            'requires_y': True,
            'pairwise': False,
            'preserves_dtype': [np.float64, np.float32]
        }

class WrappedRandomForestClassifier(RandomForestClassifier, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator_type = "classifier"

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        return self

    @classmethod
    def _more_tags(cls):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'allow_nan': True,
            'stateless': False,
            'binary_only': False,
            'requires_fit': True,
            'requires_y': True,
            'pairwise': False,
            'preserves_dtype': [np.float64, np.float32]
        }
    
@dataclass
class SortStrategy(Enum):
    AUTO = "auto"
    PARALLEL = "parallel"
    MEMORY_EFFICIENT = "memory_efficient"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    STREAM = "stream"
    BLOCK_SORT = "block_sort"
    BUCKET_SORT = "bucket_sort"
    RADIX_SORT = "radix_sort"
    COMPRESSION_SORT = "compression_sort"
    EXTERNAL_SORT = "external_sort"
    COUNTING_SORT = "counting_sort"
    LAZY_SORT = "lazy_sort"
    SEQUENTIAL_SORT = "sequential_sort"
    MICRO_SORT = "micro_sort"
    HYBRID_COMPRESSION_SORT = "hybrid_compression_sort"
    HOT_SWAP_SORT = "hot_swap_sort"
    STREAMING_HYBRID_SORT = "streaming_hybrid_sort"
    SHELL_SORT = "shell_sort"
    COMB_SORT = "comb_sort"
    PANCAKE_SORT = "pancake_sort"
    GNOME_SORT = "gnome_sort"
    CYCLE_SORT = "cycle_sort"
    BITONIC_SORT = "bitonic_sort"
    ODD_EVEN_SORT = "odd_even_sort"
    STOOGE_SORT = "stooge_sort"
    SMOOTH_SORT = "smooth_sort"

    @classmethod
    def from_str(cls, strategy: str):
        return cls[strategy.upper()]

class MultiLayerCache:
    def __init__(self, l1_size: int, l2_size: int):
        self.l1_cache = LRUCache(l1_size)
        self.l2_cache = LRUCache(l2_size)

class Algorithm(Enum):
    QUICKSORT = "quicksort"
    MERGESORT = "mergesort"
    HEAPSORT = "heapsort"
    TIMSORT = "timsort"
    INTROSORT = "introsort"
    RADIXSORT = "radixsort"
    EXTERNALMERGESORT = "externalmergesort"
    COUNTINGSORT = "countingsort"
    QUICKSELECT = "quickselect"
    INSERTIONSORT = "insertionsort"
    NONE = "none"

class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def insert(root, key):
    if root is None:
        return TreeNode(key)
    else:
        if root.val < key:
            root.right = insert(root.right, key)
        else:
            root.left = insert(root.left, key)
    return root

def inorder_traversal(root, res):
    if root:
        inorder_traversal(root.left, res)
        res.append(root.val)
        inorder_traversal(root.right, res)

def _tree_sort(self, arr: npt.NDArray) -> npt.NDArray:
    if len(arr) == 0:
        return arr

    root = TreeNode(arr[0])
    for i in range(1, len(arr)):
        insert(root, arr[i])

    sorted_arr = []
    inorder_traversal(root, sorted_arr)
    return np.array(sorted_arr)

@dataclass
class PerformanceMetrics:
    cpu_time: float = 0.0
    wall_time: float = 0.0
    memory_peak: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    thread_count: int = 0
    context_switches: int = 0
    io_operations: int = 0
    network_usage: float = 0.0
    disk_io: float = 0.0
    cache_efficiency: float = 0.0
    compression_ratio: float = 1.0


class AdaptiveCache:
    def __init__(self, initial_size: int = 1000):
        self.cache = {}
        self.size = initial_size
        self.hits = 0
        self.misses = 0
        self._access_history = deque(maxlen=initial_size)
        self._resize_threshold = 0.8
        self.min_size = initial_size // 2
        self.l1_cache = {}
        self.l2_cache = {}

    def _should_resize(self) -> bool:
        if not self._access_history:
            return False
        hit_rate = self.hits / (self.hits + self.misses + 1)
        return hit_rate < self._resize_threshold or len(self._access_history) < self.min_size

    def resize(self):
        if self._should_resize():
            self.size = max(self.min_size, int(self.size * 1.5))
            self._access_history = deque(maxlen=self.size)

    @lru_cache(maxsize=256)
    def get(self, key: Union[int, str]) -> Any:
        if key in self.l1_cache:
            self.hits += 1
            self._access_history.append(key)
            return self.l1_cache[key]
        elif key in self.l2_cache:
            self.hits += 1
            self._access_history.append(key)
            return self.l2_cache[key]
        self.misses += 1
        self.resize()
        return None

    def put(self, key: Union[int, str], value: Any) -> None:
        if len(self.l1_cache) >= self.size:
            oldest = self._access_history.popleft()
            self.l1_cache.pop(oldest, None)
        self.l1_cache[key] = value
        self._access_history.append(key)
        if len(self.l2_cache) >= self.size * 2:
            self.l2_cache.pop(next(iter(self.l2_cache)))
        self.l2_cache[key] = value

class BlockManager:
    def __init__(self, block_size: int = 4096):
        self.block_size = block_size
        self.blocks = []

    def split_into_blocks(self, arr: npt.NDArray) -> List[npt.NDArray]:
        return np.array_split(arr, max(1, len(arr) // self.block_size))

    def merge_blocks(self, blocks: List[npt.NDArray]) -> npt.NDArray:
        if not blocks:
            return np.array([])

        result = np.zeros(sum(len(block)
                          for block in blocks), dtype=blocks[0].dtype)
        pos = 0

        for block in blocks:
            result[pos:pos + len(block)] = block
            pos += len(block)

        return result

def tune_xgboost_model(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    model = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

@dataclass
class SortStats:
    execution_time: float
    memory_usage: float
    items_processed: int
    cpu_usage: float
    bucket_distribution: List[int]
    strategy_used: str
    algorithm_used: str
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    stream_chunks: int = 0
    compression_ratio: float = 1.0
    error_detected: bool = False
    fallback_strategy: str = "None"


class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._access_history = deque(maxlen=max_size)

    @lru_cache(maxsize=128)
    def get(self, key: Union[int, str]) -> Any:
        if key in self.cache:
            self.hits += 1
            self._access_history.append(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: Union[int, str], value: Any) -> None:
        if len(self.cache) >= self.max_size:
            oldest = self._access_history.popleft()
            self.cache.pop(oldest, None)
        self.cache[key] = value
        self._access_history.append(key)


class StreamProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.buffer = deque(maxlen=chunk_size)
        self._lock = threading.Lock()
        self.chunks_processed = 0
        self.linear_model = LinearRegression()
        self.last_chunk_size = chunk_size
        self.chunk_history = deque(maxlen=5)

    def process_stream(self, data_stream: Generator) -> Generator:
        for item in data_stream:
            with self._lock:
                self.buffer.append(item)
                if len(self.buffer) >= self.chunk_size:
                    self.chunks_processed += 1
                    sorted_buffer = sorted(list(self.buffer), key=lambda x: float(
                        x) if isinstance(x, (int, float, np.number)) else str(x))
                    yield sorted_buffer
                    self.chunk_history.append(len(self.buffer))
                    self.buffer.clear()
                    self._update_chunk_size()

        if self.buffer:
            self.chunks_processed += 1
            sorted_buffer = sorted(list(self.buffer), key=lambda x: float(
                x) if isinstance(x, (int, float, np.number)) else str(x))
            yield sorted_buffer
            self.chunk_history.append(len(self.buffer))

    def _update_chunk_size(self):
        if len(self.chunk_history) < 2:
            return

        x = np.arange(1, len(self.chunk_history) + 1).reshape(-1, 1)
        y = np.array(self.chunk_history)
        self.linear_model.fit(x, y)
        next_chunk_size = self.linear_model.predict(
            np.array([[len(self.chunk_history) + 1]]))[0]
        self.chunk_size = max(1000, int(next_chunk_size))


class MetricsCollector:
    def __init__(self):
        self.metrics = []
        self.disk_io_start = psutil.disk_io_counters()
        self.start_time = time.perf_counter()

    def record(self, metric_name: str, value: Any):
        self.metrics.append({
            'name': metric_name,
            'value': value,
            'timestamp': time.perf_counter() - self.start_time
        })

    def get_summary(self) -> Dict[str, Any]:
        disk_io_end = psutil.disk_io_counters()
        read_count = disk_io_end.read_bytes - self.disk_io_start.read_bytes
        write_count = disk_io_end.write_bytes - self.disk_io_start.write_bytes
        return {
            'total_duration': time.perf_counter() - self.start_time,
            'metrics': self.metrics,
            'disk_io': {
                'read_count': read_count,
                'write_count': write_count,
            }
        }


@contextmanager
def performance_tracker():
    start_time = time.perf_counter()
    start_cpu = time.process_time()

    try:
        yield
    finally:
        end_cpu = time.process_time()
        end_time = time.perf_counter()

        logger.debug(f"CPU Time: {end_cpu - start_cpu:.4f}s")
        logger.debug(f"Wall Time: {end_time - start_time:.4f}s")


def is_distributed_env():
    if os.getenv('ENV_TYPE') == "cluster":
        return True
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
        if len(ip_address.split('.')) > 3:
            return True
    except OSError:
        pass
    if psutil.cpu_count() > 4:
        return True

    return False


class EnhancedHyperionSort:
    def __init__(
        self,
        strategy: SortStrategy = SortStrategy.AUTO,
        n_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        profile: bool = False,
        cache_size: int = 2000,
        adaptive_threshold: float = 0.8,
        stream_mode: bool = False,
        block_size: int = 4096,
        use_ml_prediction: bool = True,
        compression_threshold: int = 10000,
        external_sort_threshold=100000,
        duplicate_threshold=0.5,
        eco_mode=False,
        priority_mode="speed",
        deduplicate_sort=False,
        service_mode=False,
        data_type="number",
        log_level=logging.INFO,
        benchmark=False,
        data_distribution_test=False,
        distributed=False
    ):
        self.strategy = strategy
        self.profile = profile
        self.benchmark = benchmark
        self.data_distribution_test = data_distribution_test
        self.distributed = distributed
        self.n_workers = n_workers or max(1, psutil.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.cache = AdaptiveCache(cache_size)
        self.adaptive_threshold = adaptive_threshold
        self.stream_mode = stream_mode
        self.block_manager = BlockManager(block_size)
        self.metrics = MetricsCollector()
        self._setup_logging(log_level)
        self.start_time = time.perf_counter()
        self.stream_processor = StreamProcessor(
            chunk_size=self.chunk_size or 1000)
        self.use_ml_prediction = use_ml_prediction
        self.ml_model_path = "ml_model.pkl"
        self.models = self._load_ml_models() if use_ml_prediction else []
        self.compression_threshold = compression_threshold
        self.fallback_strategy = Algorithm.MERGESORT
        self.external_sort_threshold = external_sort_threshold
        self.buffer_size = 4096
        self.duplicate_threshold = duplicate_threshold
        self.eco_mode = eco_mode
        self.priority_mode = priority_mode
        self.historical_runs = {}
        self.deduplicate_sort = deduplicate_sort
        self.service_mode = service_mode
        self.data_type = data_type
        self.comparator = self._get_default_comparator()
        self.cpu_load_data = deque(maxlen=10)
        self.load_balancer_enabled = True
        self.cache = AdaptiveCache(cache_size)
        self.historical_runs = {}
        
    def _setup_metrics(self) -> Dict[str, Any]:
        return {
            'sort_times': [],
            'memory_usage': [],
            'cache_stats': {'hits': 0, 'misses': 0},
            'block_stats': {'splits': 0, 'merges': 0}
        }
        
    @staticmethod
    def train_predict_label(feature_sample: Dict[str, Any]) -> str:
        is_nearly_sorted = feature_sample.get("is_nearly_sorted", False)
        std_dev = feature_sample.get("std_dev", 0)
        range_size = feature_sample.get("range_size", 1)
        n = feature_sample.get("n", 1)
        data_skewness = feature_sample.get("data_skewness", 0)
        data_kurtosis = feature_sample.get("data_kurtosis", 0)
        data_type = feature_sample.get("data_type", "number")

        if data_type != "number":
            return SortStrategy.ADAPTIVE.value.upper()
        if data_skewness > 2 or data_kurtosis > 5:
            return SortStrategy.BUCKET_SORT.value.upper()

        if is_nearly_sorted:
            return SortStrategy.ADAPTIVE.value.upper()

        if n > 1_000_000:
            return SortStrategy.PARALLEL.value.upper()

        if std_dev < range_size / 100:
            return SortStrategy.HYBRID.value.upper()

        if n > 100_000:
            return SortStrategy.EXTERNAL_SORT.value.upper()

        if n > 10_000:
            return SortStrategy.SEQUENTIAL_SORT.value.upper()

        if n > 1000:
            return SortStrategy.MICRO_SORT.value.upper()

        return SortStrategy.ADAPTIVE.value.upper()

    def _load_ml_models(self):
        if os.path.exists(self.ml_model_path):
            try:
                with open(self.ml_model_path, "rb") as f:
                    loaded_data = pickle.load(f)
                    if "models" in loaded_data:
                        models = loaded_data["models"]
                        self.logger.info("Loaded ML models from disk.")
                        return models
                    else:
                        self.logger.warning("No 'models' key found in loaded data. Training new models.")
                        return self._train_ml_models()
            except Exception as e:
                self.logger.warning(f"Error loading models from disk: {e}. Training new models.")
                return self._train_ml_models()
        else:
            self.logger.info("ML models not found on disk. Training new models.")
            return self._train_ml_models()

    def _save_ml_models(self, models):
        try:
            with open(self.ml_model_path, "wb") as f:
                pickle.dump({"models": models}, f)
            self.logger.info("Saved ML models to disk.")
        except Exception as e:
            self.logger.error(f"Error saving models to disk: {e}")

    def _tune_xgboost_model(self, X_train, y_train):
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        model = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_
    
    def _fine_tune_ml_models(self, X_train, y_train):
        param_distributions = {
            'learning_rate': uniform(0.001, 0.3),
            'max_depth': randint(3, 10),
            'n_estimators': randint(50, 500),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
        
        model = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False)
        
        random_search = RandomizedSearchCV(estimator=model, 
                                        param_distributions=param_distributions, 
                                        n_iter=50,
                                        cv=5,
                                        scoring='accuracy', 
                                        verbose=1,
                                        n_jobs=-1)
        
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    def _incremental_ml_training(self, new_data: List[Dict[str, Any]]):
        for record in new_data:
            self.training_data.append(record)
        self._train_ml_models(training_data_set=self.training_data)
    
    def _feature_importance_analysis(self, features: np.ndarray, labels: np.ndarray):
        model = RandomForestClassifier()
        model.fit(features, labels)
        importances = model.feature_importances_
        return importances

    def _simulate_data(self, size: int) -> npt.NDArray:
        return np.random.randint(0, size * 10, size=size)

    def incremental_training(self, new_data: List[Dict[str, Any]]):
        self.training_data.extend(new_data)
        self._train_ml_models(training_data_set=self.training_data)
    
    def _train_ml_models(self, training_data_set=None, n_samples_train=100000, **kwargs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Benchmark_results/benchmark_train_{timestamp}.pkl"

        train_file_names = [filename, f"benchmark_train_{timestamp}.pkl"]

        unique_strategies = sorted(strategy.name for strategy in SortStrategy)
        strategy_mapping = {name: idx for idx, name in enumerate(unique_strategies)}

        if training_data_set is None:
            training_data_set = []
        benchmark_folder = "Benchmark_results"
        if not os.path.exists(benchmark_folder):
            self.logger.warning(
                "'Benchmark_results' folder not found. "
                "Cannot train ML models without benchmark data."
            )
            return []

        benchmark_files = [
            f for f in os.listdir(benchmark_folder)
            if (f.startswith("benchmark_results_") or f.startswith("benchmark_train_")) and f.endswith(".pkl")
        ]

        if not benchmark_files:
            self.logger.warning(
                "No benchmark result files found in 'Benchmark_results' folder. "
                "Cannot train ML models without benchmark data."
            )
            return []

        loaded_files_count = 0
        skipped_files_count = 0
        skipped_records_count = 0

        for file in tqdm(benchmark_files, desc="Loading benchmark data", leave=False):
            try:
                file_path = os.path.join(benchmark_folder, file)
                with open(file_path, 'rb') as f:
                    benchmark_data = pickle.load(f)

                    if isinstance(benchmark_data, list):
                        training_data_set.extend(benchmark_data)
                        loaded_files_count += 1
                    else:
                        skipped_files_count += 1

            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")

        if loaded_files_count > 0:
            self.logger.info(f"Loaded benchmark data from {loaded_files_count} files.")
        if skipped_files_count > 0:
            self.logger.warning(f"Skipped {skipped_files_count} files because they did not contain a list.")

        if not training_data_set:
            self.logger.error("No training data available after loading benchmark results.")
            return []

        labels = []
        data = []
        used_strategies = set()

        for record in tqdm(training_data_set, desc="Preparing training data", leave=False):
            try:
                strategy_str = record.get('strategy', SortStrategy.ADAPTIVE.value)
                if isinstance(strategy_str, SortStrategy):
                    strategy_str = strategy_str.value

                strategy_name = SortStrategy.from_str(strategy_str).name
                if strategy_name in strategy_mapping:
                    labels.append(strategy_mapping[strategy_name])
                    used_strategies.add(strategy_name)
                    data.append([
                        record.get('std_dev', 0.0),
                        record.get('range_size', 1.0),
                        float(record.get('is_nearly_sorted', False)),
                        record.get('n', 1000.0),
                        record.get('data_skewness', 0.0),
                        record.get('data_kurtosis', 0.0)
                    ])
            except Exception as e:
                skipped_records_count += 1

        if skipped_records_count > 0:
            self.logger.warning(f"Skipped {skipped_records_count} records due to errors.")

        if len(data) < 2:
            raise ValueError(f"Not enough training data: {len(data)} samples")

        data = np.array(data, dtype=np.float64)
        labels = np.array(labels, dtype=np.int64)

        unique_labels = sorted(set(labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        models = []

        xgb_model = WrappedXGBClassifier(
            objective='multi:softmax',
            num_class=len(set(labels)),
            use_label_encoder=False,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=10,
            random_state=42,
            n_jobs=psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models.append(xgb_model)

        lgb_model = WrappedLGBMClassifier(
            objective='multiclass',
            num_class=len(set(labels)),
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1,
            verbose=-1,
            boosting_type='gbdt',
            random_state=42
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models.append(lgb_model)

        print("Training CatBoostClassifier...")
        catboost_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 5,
            'loss_function': 'MultiClass',
            'verbose': False,
            'random_state': 42
        }
        print("CatBoost parameters:", catboost_params)
        cat_model = WrappedCatBoostClassifier(**catboost_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        models.append(cat_model)

        rf_model_for_voting = WrappedRandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        )
        rf_model_for_voting.fit(X_train, y_train)
        models.append(rf_model_for_voting)


        for model in models:
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            self.logger.info(f"Model accuracy: {accuracy:.4f}")

        xgb_model_for_voting = WrappedXGBClassifier(
            objective='multi:softmax',
            num_class=len(set(labels)),
            use_label_encoder=False,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=10,
            n_jobs=psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1,
            random_state=42
        )
        lgb_model_for_voting = WrappedLGBMClassifier(
            objective='multiclass',
            num_class=len(set(labels)),
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1,
            verbose=-1,
            boosting_type='gbdt',
            random_state=42
        )

        cat_model_for_voting = WrappedCatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            loss_function='MultiClass',
            verbose=False,
            random_state=42
        )

        rf_model_for_voting_v2 = WrappedRandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        )


        voting_clf = VotingClassifier(
            estimators=[
                ('xgb', xgb_model_for_voting),
                ('lgb', lgb_model_for_voting),
                ('cat', cat_model_for_voting),
                ('rf', rf_model_for_voting_v2)
            ],
            voting='hard'
        )

        voting_clf.fit(X_train, y_train)

        y_pred = voting_clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        self.logger.info(f"Voting Classifier accuracy: {accuracy:.4f}")

        models = [voting_clf]

        self._save_ml_models(models)
        return models
         
    def _predict_strategy(self, arr: npt.NDArray) -> SortStrategy:
        if not self.models:
            return self._choose_optimal_strategy(arr)

        n = len(arr)
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.data_type != "number":
            is_nearly_sorted = False
        else:
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1

        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        data_skewness = skew(sample)
        data_kurtosis = kurtosis(sample)
        features = np.array([std_dev, range_size, is_nearly_sorted, n, data_skewness, data_kurtosis]).reshape(1, -1)

        predictions = [int(model.predict(features)[0]) for model in self.models]
        predicted_strategy_idx = max(set(predictions), key=predictions.count)

        strategy_mapping = {
            0: SortStrategy.ADAPTIVE,
            1: SortStrategy.BUCKET_SORT,
            2: SortStrategy.HYBRID,
            3: SortStrategy.PARALLEL,
            4: SortStrategy.RADIX_SORT,
            5: SortStrategy.COMPRESSION_SORT,
            6: SortStrategy.MICRO_SORT,
            7: SortStrategy.MEMORY_EFFICIENT,
            8: SortStrategy.STREAMING_HYBRID_SORT,
            9: SortStrategy.HOT_SWAP_SORT,
            10: SortStrategy.EXTERNAL_SORT,
            11: SortStrategy.COUNTING_SORT,
            12: SortStrategy.LAZY_SORT,
            13: SortStrategy.SEQUENTIAL_SORT
        }

        return strategy_mapping.get(predicted_strategy_idx, SortStrategy.ADAPTIVE)

    def adaptive_thread_scaling(self, arr: npt.NDArray):
        n = len(arr)
        if n < 1000:
            self.n_workers = 1
        elif n < 10000:
            self.n_workers = min(2, psutil.cpu_count() - 1)
        else:
            self.n_workers = min(4, psutil.cpu_count() - 1)
 
    def pipeline_processing(self, data_stream: Generator) -> Generator:
        for chunk in data_stream:
            yield np.sort(chunk)

    def chunk_wise_processing(self, arr: npt.NDArray) -> npt.NDArray:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_chunks = list(executor.map(np.sort, chunks))
        return self._merge_sorted_arrays(sorted_chunks)
    
    def hierarchical_parallelism(self, arr: npt.NDArray) -> npt.NDArray:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_chunks = list(executor.map(self._parallel_sort_block, chunks))
        return self._merge_sorted_arrays(sorted_chunks)
    
    def set_thread_affinity(self):
        p = psutil.Process(os.getpid())
        p.cpu_affinity([i for i in range(psutil.cpu_count())])

    def use_shared_memory(self, arr: npt.NDArray) -> npt.NDArray:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(shared_arr, arr)
        return shared_arr
   
    def feature_reduction(self, features: np.ndarray) -> np.ndarray:
        important_features = [0, 1, 2, 3]
        return features[:, important_features]

    def cross_validation(self, X_train, y_train):
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def incremental_updates(self, new_data: List[Dict[str, Any]]):
        self.training_data.extend(new_data)
        self._train_ml_models(training_data_set=self.training_data)
    
    def bagging_models(self, X_train, y_train):
        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10)
        model.fit(X_train, y_train)
        return model

    def performance_benchmarks(self, models: List):
        for model in models:
            y_pred = model.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            self.logger.info(f"Model accuracy: {accuracy:.4f}")
   
    async def async_file_access(self, file_path: str, offset: int, size: int, dtype: np.dtype) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._read_chunk_sync, file_path, offset, size, dtype))
    
    def reduce_disk_io(self, arr: npt.NDArray) -> npt.NDArray:
        compressed_data = compress(arr.tobytes())
        decompressed_arr = np.frombuffer(decompress(compressed_data), dtype=arr.dtype)
        return decompressed_arr

    def data_compression(self, arr: npt.NDArray) -> npt.NDArray:
        compressed_data = compress(arr.tobytes())
        decompressed_arr = np.frombuffer(decompress(compressed_data), dtype=arr.dtype)
        return decompressed_arr

    def task_scheduling(self, tasks: List[Callable]):
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            executor.map(lambda task: task(), tasks)

    def data_distribution_detection(self, arr: npt.NDArray) -> str:
        skewness = skew(arr)
        kurtosis = kurtosis(arr)
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return "uniform"
        else:
            return "skewed"

    async def chunked_external_sorting(self, arr: npt.NDArray) -> npt.NDArray:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))
        sorted_chunks = []
        for chunk in chunks:
            sorted_chunks.append(np.sort(chunk))
        return self._merge_sorted_arrays(sorted_chunks)

    def statistical_insights(self, arr: npt.NDArray) -> Dict[str, float]:
        return {
            "skewness": skew(arr),
            "kurtosis": kurtosis(arr)
        }

    def realtime_monitoring(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        self.logger.info(f"CPU Usage: {cpu_usage}%")
        self.logger.info(f"Memory Usage: {memory_info.percent}%")

    def dynamic_model_switching(self, arr: npt.NDArray):
        if self.data_distribution_detection(arr) == "uniform":
            self.strategy = SortStrategy.BUCKET_SORT
        else:
            self.strategy = SortStrategy.INTROSORT
        
    def _advanced_block_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)

        blocks = self.block_manager.split_into_blocks(arr)
        self.metrics.record('block_splits', 1)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_blocks = list(tqdm(executor.map(self._optimize_block_sort, blocks), total=len(blocks), desc="Sorting blocks", leave=False))

        while len(sorted_blocks) > 1:
            new_blocks = []
            for i in tqdm(range(0, len(sorted_blocks), 2), desc="Merging blocks", leave=False):
                if i + 1 < len(sorted_blocks):
                    merged = self._merge_sorted_arrays([sorted_blocks[i], sorted_blocks[i + 1]])
                    new_blocks.append(merged)
                else:
                    new_blocks.append(sorted_blocks[i])
            sorted_blocks = new_blocks
            self.metrics.record('block_merges', 1)

        return sorted_blocks[0]
    
    def _optimize_block_sort(self, block: npt.NDArray) -> npt.NDArray:
        if len(block) < 16:
            return self._insertion_sort(block)

        std_dev = np.std(block)
        range_size = np.ptp(block)

        if std_dev < range_size / 100:
            return self._bucket_sort(block)
        elif len(block) < 1000:
            return self._quicksort(block)
        else:
            return self._introsort(block)

    def _radix_sort(self, arr: npt.NDArray) -> npt.NDArray:
        max_val = int(np.max(arr))
        exp = 1
        while max_val // exp > 0:
            buckets = [[] for _ in range(10)]
            for x in arr:
                digit = int((x // exp) % 10)
                buckets[digit].append(x)
            arr = [x for bucket in buckets for x in bucket]
            exp *= 10
        return np.array(arr)

    def _bucket_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) == 0:
            return arr

        n_buckets = self._optimize_bucket_count(arr)
        min_val, max_val = arr.min(), arr.max()

        if min_val == max_val:
            return arr

        buckets = [[] for _ in range(n_buckets)]
        width = (max_val - min_val) / n_buckets

        for x in arr:
            bucket_idx = int((x - min_val) / width)
            if bucket_idx == n_buckets:
                bucket_idx -= 1
            buckets[bucket_idx].append(x)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_buckets = list(executor.map(
                lambda x: sorted(x) if len(x) > 0 else [],
                buckets
            ))

        return np.concatenate([np.array(bucket) for bucket in sorted_buckets if bucket])

    def _quicksort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) <= 16:
            return self._insertion_sort(arr)

        pivot = self._ninther(arr)
        left = arr[arr < pivot]
        middle = arr[arr == pivot]
        right = arr[arr > pivot]

        if len(arr) > 1000:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_left = executor.submit(self._quicksort, left)
                future_right = executor.submit(self._quicksort, right)
                sorted_left = future_left.result()
                sorted_right = future_right.result()
                gc.collect()

                return np.concatenate([sorted_left, middle, sorted_right])
        else:
            return np.concatenate([
                self._quicksort(left),
                middle,
                self._quicksort(right)
            ])

    def _introsort(self, arr: npt.NDArray, max_depth: Optional[int] = None) -> npt.NDArray:
        if max_depth is None:
            max_depth = 2 * int(math.log2(len(arr)))

        if len(arr) <= 16:
            return self._insertion_sort(arr)
        elif max_depth == 0:
            return self._heapsort(arr)
        else:
            pivot = self._ninther(arr)
            left = arr[arr < pivot]
            middle = arr[arr == pivot]
            right = arr[arr > pivot]

            if len(arr) > 1000:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_left = executor.submit(
                        self._introsort, left, max_depth - 1
                    )
                    future_right = executor.submit(
                        self._introsort, right, max_depth - 1
                    )
                    sorted_left = future_left.result()
                    sorted_right = future_right.result()
                    gc.collect()
                    return np.concatenate([sorted_left, middle, sorted_right])
            else:
                return np.concatenate([
                    self._introsort(left, max_depth - 1),
                    middle,
                    self._introsort(right, max_depth - 1)
                ])

    def _ninther(self, arr: npt.NDArray) -> float:
        if len(arr) < 9:
            return np.median(arr)

        thirds = len(arr) // 3
        medians = [
            np.median([arr[i], arr[i + thirds], arr[i + 2 * thirds]])
            for i in range(3)
        ]
        return np.median(medians)

    def _insertion_sort(self, arr: npt.NDArray) -> npt.NDArray:
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    def _heapsort(self, arr: npt.NDArray) -> npt.NDArray:
        def heapify(n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[left] > arr[largest]:
                largest = left

            if right < n and arr[right] > arr[largest]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(n, largest)

        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(n, i)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(i, 0)

        return arr

    def _stream_sort_and_collect(self, data_stream: Generator) -> Tuple[npt.NDArray, SortStats]:
        all_chunks = []
        for chunk in tqdm(self.stream_processor.process_stream(data_stream), desc="Processing stream", leave=False):
            all_chunks.append(np.array(chunk))

        if not all_chunks:
            return np.array([]), self._calculate_stats(np.array([]), "stream", "none", stream_chunks=0)

        merged_array = np.concatenate(all_chunks)
        return merged_array, self._calculate_stats(merged_array, "stream", "timsort", stream_chunks=self.stream_processor.chunks_processed)

    def _compression_sort(self, arr: npt.NDArray) -> Tuple[npt.NDArray, float]:
        if len(arr) < self.compression_threshold:
            return arr, 1.0

        compressed_data = zstd.compress(arr.tobytes())
        compression_ratio = len(compressed_data) / arr.nbytes

        if compression_ratio > 1.0:
            return arr, 1.0

        decompressed_arr = np.frombuffer(zstd.decompress(compressed_data), dtype=arr.dtype)
        sorted_arr = np.sort(decompressed_arr)
        return sorted_arr, compression_ratio

    def _pivot_tree_partition(self, arr: npt.NDArray) -> List[npt.NDArray]:
        if len(arr) <= 100:
            return [arr]

        def _recursive_partition(arr, depth=0):
            if len(arr) <= 100 or depth > 10:
                return [arr]

            pivot = self._ninther(arr)
            left = arr[arr < pivot]
            middle = arr[arr == pivot]
            right = arr[arr > pivot]

            return _recursive_partition(left, depth+1) + [middle] + _recursive_partition(right, depth+1)

        partitions = _recursive_partition(arr)
        return [part for part in partitions if len(part) > 0]

    async def _read_chunk(self, file_path: str, offset: int, size: int, dtype: np.dtype) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._read_chunk_sync, file_path, offset, size, dtype))

    def _read_chunk_sync(self, file_path: str, offset: int, size: int, dtype: np.dtype) -> np.ndarray:
        with open(file_path, 'rb') as file:
            try:
                file.seek(offset)
                data = file.read(size - (size % dtype.itemsize))
                return np.frombuffer(data, dtype=dtype)
            except Exception as e:
                self.logger.warning(
                    f"Unable to read chunk from the file {file_path} with code {e}")
                return np.array([], dtype=dtype)

    async def _external_sort(self, arr: npt.NDArray) -> npt.NDArray:
        file_path = "temp_data.bin"
        dtype = arr.dtype
        arr.tofile(file_path)
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        num_chunks = math.ceil(len(arr) * arr.itemsize / chunk_size)

        with open(file_path, 'wb') as file:
            file.write(b'\0' * arr.nbytes)

        with open(file_path, "r+b") as file:
            mem = mmap.mmap(file.fileno(), 0)
            mem[:arr.nbytes] = arr.tobytes()

            sorted_chunks = []
            tasks = []
            for i in tqdm(range(num_chunks), desc="Reading chunks", leave=False):
                offset = i * chunk_size
                size = min(chunk_size, len(arr) * arr.itemsize - offset)

                task = self._read_chunk(file_path, offset, size, dtype)
                tasks.append(task)

            chunks = await asyncio.gather(*tasks)

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                sorted_chunks = list(tqdm(executor.map(np.sort, chunks), total=len(chunks), desc="Sorting chunks", leave=False))

            if len(sorted_chunks) == 0:
                return np.array([])
            result = self._multi_way_merge(sorted_chunks)

        try:
            mem.close()
            file.close()
            os.unlink(file_path)
        except Exception as e:
            self.logger.error(f"Unable to close or unlink the file: {file_path}, error {e}")
        gc.collect()
        return result

    def _multi_way_merge(self, sorted_chunks: List[npt.NDArray]) -> npt.NDArray:
        merged = []
        heap = []

        for i, arr in enumerate(sorted_chunks):
            if len(arr) > 0:
                heapq.heappush(heap, (arr[0], i, 0))

        while heap:
            val, arr_idx, elem_idx = heapq.heappop(heap)
            merged.append(val)

            if elem_idx + 1 < len(sorted_chunks[arr_idx]):
                next_val = sorted_chunks[arr_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))

        return np.array(merged)

    def _quickselect(self, arr: npt.NDArray, k: int) -> npt.NDArray:
        if k < 1 or k > len(arr):
            raise ValueError("k is out of bounds")

        def _select(arr, left, right, k):
            if left == right:
                return arr[left]

            pivot_index = random.randint(left, right)
            pivot = arr[pivot_index]
            arr[pivot_index], arr[right] = arr[right], arr[pivot_index]

            store_index = left
            for i in range(left, right):
                if arr[i] < pivot:
                    arr[store_index], arr[i] = arr[i], arr[store_index]
                    store_index += 1

            arr[store_index], arr[right] = arr[right], arr[store_index]

            if k == store_index + 1:
                return arr[store_index]
            elif k < store_index + 1:
                return _select(arr, left, store_index - 1, k)
            else:
                return _select(arr, store_index + 1, right, k)

        return _select(arr, 0, len(arr) - 1, k)

    def _lazy_sort(self, arr: npt.NDArray, k: int, top: bool = False) -> npt.NDArray:
        if k >= len(arr):
            return np.sort(arr)

        if top:
            result = self._quickselect(arr, len(arr) - k + 1)
            if k < len(arr):
                result_arr = arr[arr >= result]
            else:
                result_arr = arr
        else:
            result = self._quickselect(arr, k)
            if k > 1:
                result_arr = arr[arr <= result]
            else:
                result_arr = arr
        return result_arr

    def _sequential_smart_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) <= 16:
            return self._insertion_sort(arr)

        sample_size = min(1000, len(arr))
        sample = arr[np.random.choice(len(arr), sample_size, replace=False)]
        if self.data_type != "number":
            return np.sort(arr)
        is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1

        if is_nearly_sorted:
            return np.sort(arr, kind='stable')
        else:
            return self._introsort(arr)

    def _pre_sort_compression(self, arr: npt.NDArray) -> Tuple[npt.NDArray, float]:
        if len(arr) < self.compression_threshold:
            return arr, 1.0

        compressed_data = compress(arr.tobytes())
        compression_ratio = len(compressed_data) / arr.nbytes

        if compression_ratio >= 1.0:
            return arr, 1.0

        decompressed_arr = np.frombuffer(
            decompress(compressed_data), dtype=arr.dtype)
        return np.sort(decompressed_arr), compression_ratio

    def _fallback_strategy(self, arr: npt.NDArray, failed_strategy: SortStrategy) -> Tuple[npt.NDArray, SortStats]:
        self.logger.warning(
            f"Fallback strategy initiated from {failed_strategy.value} to {self.fallback_strategy.value}.")
        if self.fallback_strategy == Algorithm.MERGESORT:
            result = np.sort(arr)
            return result, self._calculate_stats(arr, "fallback", self.fallback_strategy.value, error_detected=True)
        else:
            self.logger.warning(
                "Fallback strategy failed, returning unsorted array.")
            return arr, self._calculate_stats(arr, "fallback_failed", "none", error_detected=True)

    def _analyze_duplicate_ratio(self, arr: npt.NDArray) -> float:
        if len(arr) == 0:
            return 0.0
        unique_count = len(np.unique(arr))
        return 1.0 - (unique_count / len(arr))

    def _validate_data(self, arr: npt.NDArray) -> bool:
        if self.data_type == "number":
            if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
                self.logger.error(
                    "Invalid data detected (NaN, infinite or not a number).")
                return False
            if np.issubdtype(arr.dtype, np.integer) and np.any(arr < 0):
                self.logger.error(
                    "Invalid data detected (negative numbers in integer array).")
                return False
        return True

    def _process_mixed_data(self, arr: list) -> np.ndarray:
        if not arr:
            return np.array([])

        if self.data_type == "number":
            try:
                return np.array(arr, dtype=np.float64)
            except ValueError as e:
                self.logger.error(
                    f"Cannot process mixed numeric data, falling back to string sort: {e}")
                self.data_type = "string"
                return self._process_mixed_data(arr)

        elif self.data_type == "string":
            return np.array([str(item) for item in arr], dtype=str)

        elif self.data_type == "object":
            if not all(isinstance(x, dict) for x in arr):
                self.logger.error(f"Object must be a dict type")
                return np.array([])

            keys = set()
            for item in arr:
                keys.update(item.keys())
            keys = list(keys)

            if len(keys) == 0:
                return np.array(arr, dtype=object)

            if len(keys) > 1:
                self.logger.warning(
                    "Object has more than one key, only the first key will be used.")

            key = keys[0]
            processed_arr = []
            for item in arr:
                if isinstance(item, dict):
                    if key in item:
                        processed_arr.append(item[key])
                    else:
                        self.logger.warning(
                            f"Object missing key '{key}', appending None.")
                        processed_arr.append(None)
                else:
                    processed_arr.append(item)
            return np.array(processed_arr, dtype=object)
        else:
            self.logger.error(f"Invalid data type {self.data_type}")
            return np.array([])

    def _get_default_comparator(self):
        if self.data_type == "number":
            return lambda a, b: (a > b) - (a < b)
        elif self.data_type == "string":
            return lambda a, b: (len(a) > len(b)) - (len(a) < len(b)) if len(a) != len(b) else (a > b) - (a < b)
        elif self.data_type == "object":
            return lambda a, b: (a > b) - (a < b)
        else:
            self.logger.error("Invalid data type for comparator.")
            return lambda a, b: 0

    def _dynamic_comparator(self, a: Any, b: Any) -> int:
        return self.comparator(a, b)

    def _parallel_compare(self, arr: npt.NDArray, indices: List[Tuple[int, int]]) -> List[int]:
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(lambda idx: self._dynamic_comparator(
                arr[idx[0]], arr[idx[1]]), indices))
            return results

    def _deduplicate(self, arr: npt.NDArray) -> npt.NDArray:
        return np.unique(arr)

    def _micro_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 100:
            return self._insertion_sort(arr)

        chunk_size = max(1, int(math.sqrt(len(arr))))
        chunks = [arr[i:i + chunk_size]
                  for i in range(0, len(arr), chunk_size)]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_chunks = list(executor.map(self._insertion_sort, chunks))

        return self._merge_sorted_arrays(sorted_chunks)

    def _hybrid_compression_sort(self, arr: npt.NDArray) -> Tuple[npt.NDArray, float]:
        if len(arr) < self.compression_threshold:
            return arr, 1.0

        compressed_data = compress(arr.tobytes())
        compression_ratio = len(compressed_data) / arr.nbytes

        if compression_ratio >= 1.0:
            return arr, 1.0

        decompressed_arr = np.frombuffer(
            decompress(compressed_data), dtype=arr.dtype)

        chunk_size = self._adaptive_chunk_size(
            len(decompressed_arr), decompressed_arr.itemsize)
        chunks = np.array_split(decompressed_arr, max(
            1, len(decompressed_arr) // chunk_size))

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_chunks = list(executor.map(np.sort, chunks))

        result = self._merge_sorted_arrays(sorted_chunks)
        return result, compression_ratio

    def _compress_repeating_values(self, arr: npt.NDArray) -> Tuple[bytes, dict, np.dtype]:
        if len(arr) == 0:
            return b"", {}, arr.dtype

        value_counts = {}
        for item in arr:
            if item in value_counts:
                value_counts[item] += 1
            else:
                value_counts[item] = 1

        threshold = max(1, len(arr) // 100)

        compressed_dict = {k: v for k,
                           v in value_counts.items() if v > threshold}

        compressed_indices = []
        compressed_data = []
        index_mapping = {}

        for i, item in enumerate(arr):
            if item in compressed_dict:
                if item not in index_mapping:
                    index_mapping[item] = len(index_mapping)
                compressed_indices.append(index_mapping[item])
            else:
                compressed_data.append(item)

        return pickle.dumps(compressed_indices), compressed_dict, arr.dtype

    def _decompress_repeating_values(self, compressed_indices: bytes, compressed_dict: dict, original_dtype: np.dtype) -> npt.NDArray:

        if not compressed_indices or not compressed_dict:
            return np.array([], dtype=original_dtype)

        compressed_indices = pickle.loads(compressed_indices)

        original_arr = []
        reverse_mapping = {v: k for k, v in compressed_dict.items()}
        for index in compressed_indices:
            original_arr.append(reverse_mapping.get(index))

        return np.array(original_arr, dtype=original_dtype)

    def _predict_data_distribution(self, arr: npt.NDArray) -> str:
        std_dev = np.std(arr)
        range_size = np.ptp(arr)
        skewness = skew(arr)
        kurtosis = kurtosis(arr)
        
        if std_dev < range_size / 10:
            return "uniform"
        elif abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return "gaussian"
        else:
            return "unknown"
    
    def _parallel_block_merge(self, sorted_blocks: List[npt.NDArray]) -> npt.NDArray:
        heap = []
        for i, block in enumerate(sorted_blocks):
            if len(block) > 0:
                heapq.heappush(heap, (block[0], i, 0))
        
        merged = []
        while heap:
            val, block_idx, elem_idx = heapq.heappop(heap)
            merged.append(val)
            if elem_idx + 1 < len(sorted_blocks[block_idx]):
                next_val = sorted_blocks[block_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, block_idx, elem_idx + 1))
        
        return np.array(merged)

    def use_memmap(self, arr: npt.NDArray, filename: str) -> npt.NDArray:
        memmap_arr = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=arr.shape)
        np.copyto(memmap_arr, arr)
        return memmap_arr

    def dynamic_memory_allocation(self):
        if self.cache.hits / (self.cache.hits + self.cache.misses) < self.adaptive_threshold:
            self.cache.size = min(self.cache.size * 2, 10000)
           
    async def _pipelined_external_sort(self, arr: np.ndarray) -> np.ndarray:
        file_path = "temp_data.bin"
        dtype = arr.dtype
        arr.tofile(file_path)
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        num_chunks = math.ceil(len(arr) * arr.itemsize / chunk_size)

        with open(file_path, 'wb') as file:
            file.write(b'\0' * arr.nbytes)

        with open(file_path, "r+b") as file:
            mem = mmap.mmap(file.fileno(), 0)
            mem[:arr.nbytes] = arr.tobytes()

            sorted_chunks = []
            tasks = []
            for i in tqdm(range(num_chunks), desc="Reading chunks", leave=False):
                offset = i * chunk_size
                size = min(chunk_size, len(arr) * arr.itemsize - offset)

                task = self._read_chunk(file_path, offset, size, dtype)
                tasks.append(task)

            chunks = await asyncio.gather(*tasks)

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                sorted_chunks = list(tqdm(executor.map(np.sort, chunks), total=len(chunks), desc="Sorting chunks", leave=False))

            if len(sorted_chunks) == 0:
                return np.array([])
            result = self._merge_sorted_arrays(sorted_chunks)

        try:
            mem.close()
            file.close()
            os.unlink(file_path)
        except Exception as e:
            self.logger.error(f"Unable to close or unlink the file: {file_path}, error {e}")
        gc.collect()
        return result

    def _cache_best_strategy(self, arr: npt.NDArray, strategy: SortStrategy):
        key = self._create_historical_key(arr)
        self.historical_runs[key] = strategy

    def _pre_sort_sampling(self, arr: npt.NDArray) -> SortStrategy:
        sample_size = min(1000, len(arr))
        sample = arr[np.random.choice(len(arr), sample_size, replace=False)]
        return self._predict_strategy(sample)
    
    def _predictive_thread_scaling(self, arr: npt.NDArray):
        n = len(arr)
        if n < 1000:
            self.n_workers = 1
        elif n < 10000:
            self.n_workers = min(2, psutil.cpu_count() - 1)
        else:
            self.n_workers = min(4, psutil.cpu_count() - 1)
        self.logger.info(f"Adjusted number of workers to {self.n_workers}")
    
    def _batch_sort_streaming(self, data_stream: Generator) -> Tuple[npt.NDArray, SortStats]:
        batch_size = 1000
        batches = []
        for chunk in data_stream:
            batches.append(chunk)
            if len(batches) >= batch_size:
                sorted_batch = np.sort(np.concatenate(batches))
                yield sorted_batch
                batches = []
        if batches:
            sorted_batch = np.sort(np.concatenate(batches))
            yield sorted_batch

    def _dynamic_fallback_strategy(self, arr: npt.NDArray, failed_strategy: SortStrategy) -> Tuple[npt.NDArray, SortStats]:
        self.logger.warning(f"Fallback strategy initiated from {failed_strategy.value} to {self.fallback_strategy.value}.")
        if self.fallback_strategy == Algorithm.MERGESORT:
            result = np.sort(arr)
            return result, self._calculate_stats(arr, "fallback", self.fallback_strategy.value, error_detected=True)
        else:
            self.logger.warning("Fallback strategy failed, returning unsorted array.")
            return arr, self._calculate_stats(arr, "fallback_failed", "none", error_detected=True)
    
    def _smart_memory_management(self):
        gc.collect()
        self.logger.info("Garbage collection completed.")
    
    def _index_sort(self, arr: npt.NDArray) -> npt.NDArray:
        indices = np.argsort(arr)
        return arr[indices]

    def _multi_dimensional_sort(self, arr: npt.NDArray) -> npt.NDArray:
        hilbert_curve = HilbertCurve(p=16, n=arr.shape[1])
        hilbert_indices = np.array([hilbert_curve.distance_from_coordinates(coord) for coord in arr])
        sorted_indices = np.argsort(hilbert_indices)
        return arr[sorted_indices]

    def _trie_sort(self, arr: List[str]) -> List[str]:
        from datrie import Trie
        trie = Trie(ranges=[chr(i) for i in range(32, 127)])
        for word in arr:
            trie[word] = word
        return list(trie.values())

    def _topological_sort(self, graph: Dict[Any, List[Any]]) -> List[Any]:
        from collections import deque
        in_degree = {u: 0 for u in graph}
        for u in graph:
            for v in graph[u]:
                in_degree[v] += 1
        queue = deque([u for u in graph if in_degree[u] == 0])
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return topo_order

    def _hierarchical_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)
        partitions = self._advanced_partition(arr)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_partitions = list(executor.map(np.sort, partitions))
        return np.concatenate(sorted_partitions)

    def _thread_pinning(self):
        import os
        import psutil
        p = psutil.Process(os.getpid())
        p.cpu_affinity([i for i in range(psutil.cpu_count())])
        self.logger.info("Thread pinning applied.")
    
    def _adaptive_hybrid_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)
        partitions = self._advanced_partition(arr)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_partitions = list(executor.map(np.sort, partitions))
        return np.concatenate(sorted_partitions)

    def _compress_and_merge_sort(self, arr: npt.NDArray) -> npt.NDArray:
        compressed_data = compress(arr.tobytes())
        decompressed_arr = np.frombuffer(decompress(compressed_data), dtype=arr.dtype)
        sorted_arr = np.sort(decompressed_arr)
        return sorted_arr

    def _dynamic_cpu_load_balancing(self, arr: npt.NDArray):
        cpu_load = psutil.cpu_percent()
        self.cpu_load_data.append(cpu_load)

        avg_load = np.mean(self.cpu_load_data) if self.cpu_load_data else cpu_load

        if avg_load > 80 and self.n_workers > 1:
            self.n_workers = max(1, self.n_workers - 1)
            self.logger.warning(f"CPU overloaded, reducing workers to {self.n_workers}")
        elif avg_load < 30 and self.n_workers < psutil.cpu_count() - 1 and len(arr) > 1_000_000:
            self.n_workers = min(psutil.cpu_count() - 1, self.n_workers + 1)
            self.logger.info(f"CPU underutilized, increasing workers to {self.n_workers}")

    def _parallel_pipeline_sort(self, arr: npt.NDArray) -> npt.NDArray:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))

        sorted_chunks = []

        def sort_chunk(chunk):
            sorted_chunk = np.sort(chunk)
            sorted_chunks.append(sorted_chunk)
            return sorted_chunk

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            executor.map(sort_chunk, chunks)

        return self._merge_sorted_arrays(sorted_chunks)
    
    def _downsample_sort(self, arr: npt.NDArray, factor: int = 10) -> npt.NDArray:
        if len(arr) < factor:
            return np.sort(arr)

        downsampled = arr[::factor]
        sorted_downsampled = np.sort(downsampled)
        return sorted_downsampled
    
    def _intelligent_parallelization(self, arr: npt.NDArray):
        n = len(arr)
        if n < 1000:
            self.n_workers = 1
        elif n < 10000:
            self.n_workers = min(2, psutil.cpu_count() - 1)
        else:
            self.n_workers = min(4, psutil.cpu_count() - 1)
        self.logger.info(f"Adjusted number of workers to {self.n_workers}")
    
    def _priority_based_block_sort(self, arr: npt.NDArray, priority_indices: List[int]) -> npt.NDArray:
        blocks = self.block_manager.split_into_blocks(arr)
        priority_blocks = [blocks[i] for i in priority_indices]
        non_priority_blocks = [blocks[i] for i in range(len(blocks)) if i not in priority_indices]

        sorted_priority_blocks = [np.sort(block) for block in priority_blocks]
        sorted_non_priority_blocks = [np.sort(block) for block in non_priority_blocks]

        return self.block_manager.merge_blocks(sorted_priority_blocks + sorted_non_priority_blocks)
    
    def _repetition_compression(self, arr: npt.NDArray) -> Tuple[npt.NDArray, float]:
        unique, counts = np.unique(arr, return_counts=True)
        compression_ratio = len(unique) / len(arr)
        compressed_arr = np.repeat(unique, counts)
        return compressed_arr, compression_ratio
    
    def _dry_run_performance_check(self, arr: npt.NDArray) -> SortStrategy:
        sample_size = min(1000, len(arr))
        sample = arr[np.random.choice(len(arr), sample_size, replace=False)]
        return self._predict_strategy(sample)
    
    def _concurrent_combined_sorting(self, arr: npt.NDArray) -> npt.NDArray:
        strategies = [
            SortStrategy.QUICKSORT,
            SortStrategy.MERGESORT,
            SortStrategy.HEAPSORT,
            SortStrategy.TIMSORT
        ]
        results = []

        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = [executor.submit(self._benchmark_on_the_fly, arr, strategy) for strategy in strategies]
            for future in futures:
                results.append(future.result())

        best_result = min(results, key=lambda x: x[1].execution_time)
        return best_result[0]
   
    def _advanced_fallback_manager(self, arr: npt.NDArray, failed_strategy: SortStrategy) -> Tuple[npt.NDArray, SortStats]:
        fallback_strategies = [
            SortStrategy.MERGESORT,
            SortStrategy.HEAPSORT,
            SortStrategy.TIMSORT
        ]
        for strategy in fallback_strategies:
            try:
                result, stats = self._sort_with_fallback(arr, strategy)
                if stats.error_detected:
                    continue
                return result, stats
            except Exception as e:
                self.logger.error(f"Fallback strategy {strategy.value} failed: {e}")
        return arr, self._calculate_stats(arr, "fallback_failed", "none", error_detected=True)
     
    def _dynamic_performance_monitoring(self, arr: npt.NDArray):
        start_time = time.perf_counter()
        sorted_arr = self._adaptive_sort(arr)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        self.metrics.record('dynamic_performance', execution_time)
        self.logger.info(f"Dynamic performance monitoring: {execution_time:.4f}s")
     
    def _real_time_forecasting(self, arr: npt.NDArray) -> Dict[str, Any]:
        n = len(arr)
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.use_ml_prediction:
            predicted_strategy = self._predict_strategy(sample)
        else:
            predicted_strategy = self._choose_optimal_strategy(sample)

        estimated_time = 0.00001 * n + 0.000001 * n**2
        estimated_memory = n * arr.itemsize

        return {
            "estimated_time": f"{estimated_time:.4f}s",
            "estimated_memory": f"{estimated_memory / (1024 * 1024):.2f} MB",
            "suggested_strategy": predicted_strategy
        }
           
    def _data_pipelining_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)

        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))

        sorted_chunks = []

        def sort_chunk(chunk):
            sorted_chunk = np.sort(chunk)
            sorted_chunks.append(sorted_chunk)
            return sorted_chunk

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            executor.map(sort_chunk, chunks)

        return self._merge_sorted_arrays(sorted_chunks)

    def _detect_homogeneous_data(self, arr: npt.NDArray) -> bool:
        return np.all(arr == arr[0])

    def _cluster_sort(self, arr: npt.NDArray) -> npt.NDArray:
        n_clusters = min(10, len(arr) // 100)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(arr.reshape(-1, 1))
        sorted_arr = np.concatenate([np.sort(arr[labels == i]) for i in range(n_clusters)])
        return sorted_arr

    def _smart_block_merge(self, blocks: List[npt.NDArray]) -> npt.NDArray:
        if not blocks:
            return np.array([])

        result = np.zeros(sum(len(block)
                        for block in blocks), dtype=blocks[0].dtype)
        pos = 0

        for block in blocks:
            result[pos:pos + len(block)] = block
            pos += len(block)

        return result

    def _multi_stage_sort(self, arr: npt.NDArray) -> npt.NDArray:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_chunks = list(executor.map(np.sort, chunks))

        return self._merge_sorted_arrays(sorted_chunks)

    def _shared_memory_sort(self, arr: npt.NDArray) -> npt.NDArray:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(shared_arr, arr)

        sorted_arr = np.sort(shared_arr)
        shm.close()
        shm.unlink()
        return sorted_arr

    def _range_based_partition(self, arr: npt.NDArray) -> List[npt.NDArray]:
        n_buckets = self._optimize_bucket_count(arr)
        min_val, max_val = arr.min(), arr.max()

        if min_val == max_val:
            return [arr]

        buckets = [[] for _ in range(n_buckets)]
        width = (max_val - min_val) / n_buckets

        for x in arr:
            bucket_idx = int((x - min_val) / width)
            if bucket_idx == n_buckets:
                bucket_idx -= 1
            buckets[bucket_idx].append(x)

        return [np.array(bucket) for bucket in buckets if bucket]

    def _adaptive_algorithm_tuning(self, arr: npt.NDArray) -> SortStrategy:
        n = len(arr)
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.data_type != "number":
            is_nearly_sorted = False
        else:
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1

        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        data_skewness = skew(sample)
        data_kurtosis = kurtosis(sample)

        if data_skewness > 2 or data_kurtosis > 5:
            return SortStrategy.BUCKET_SORT

        if is_nearly_sorted:
            return SortStrategy.ADAPTIVE

        if n > 1_000_000:
            return SortStrategy.PARALLEL

        if std_dev < range_size / 100:
            return SortStrategy.HYBRID

        if n > 100_000:
            return SortStrategy.EXTERNAL_SORT

        if n > 10_000:
            return SortStrategy.SEQUENTIAL_SORT

        if n > 1000:
            return SortStrategy.MICRO_SORT

        return SortStrategy.ADAPTIVE

    def _simulate_sorting_strategies(self, arr: npt.NDArray) -> SortStrategy:
        strategies = [
            SortStrategy.QUICKSORT,
            SortStrategy.MERGESORT,
            SortStrategy.HEAPSORT,
            SortStrategy.TIMSORT,
            SortStrategy.INTROSORT,
            SortStrategy.RADIXSORT,
            SortStrategy.EXTERNALMERGESORT,
            SortStrategy.COUNTINGSORT,
            SortStrategy.QUICKSELECT,
            SortStrategy.INSERTIONSORT
        ]
        best_strategy = SortStrategy.QUICKSORT
        best_time = float('inf')

        for strategy in strategies:
            start_time = time.perf_counter()
            self._benchmark_on_the_fly(arr, strategy)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if elapsed_time < best_time:
                best_time = elapsed_time
                best_strategy = strategy

        return best_strategy

    def _ensemble_prediction(self, features: np.ndarray) -> SortStrategy:
        predictions = [model.predict(features)[0] for model in self.models]
        predicted_strategy_idx = max(set(predictions), key=predictions.count)
        strategy_mapping = {
            0: SortStrategy.ADAPTIVE,
            1: SortStrategy.BUCKET_SORT,
            2: SortStrategy.HYBRID,
            3: SortStrategy.PARALLEL,
            4: SortStrategy.RADIX_SORT,
            5: SortStrategy.COMPRESSION_SORT,
            6: SortStrategy.MICRO_SORT,
            7: SortStrategy.MEMORY_EFFICIENT,
            8: SortStrategy.STREAMING_HYBRID_SORT,
            9: SortStrategy.HOT_SWAP_SORT,
            10: SortStrategy.EXTERNAL_SORT,
            11: SortStrategy.COUNTING_SORT,
            12: SortStrategy.LAZY_SORT,
            13: SortStrategy.SEQUENTIAL_SORT
        }
        return strategy_mapping.get(predicted_strategy_idx, SortStrategy.ADAPTIVE)
 
    def _save_run_history(self, arr: npt.NDArray, strategy: SortStrategy, stats: SortStats):
        key = self._create_historical_key(arr)
        self.historical_runs[key] = {
            'strategy': strategy,
            'stats': stats
        }

    def _load_run_history(self, arr: npt.NDArray) -> Optional[Dict[str, Any]]:
        key = self._create_historical_key(arr)
        return self.historical_runs.get(key)
    
    def _nested_parallel_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)

        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))

        with ThreadPoolExecutor(max_workers=self.n_workers) as main_executor:
            sorted_chunks = list(main_executor.map(
                self._parallel_sort_block, chunks))

        return self._merge_sorted_arrays(sorted_chunks)

    def _parallel_sort_block(self, block: npt.NDArray) -> npt.NDArray:
        sub_chunk_size = self._adaptive_chunk_size(len(block), block.itemsize)
        sub_chunks = np.array_split(
            block, max(1, len(block) // sub_chunk_size))

        with ThreadPoolExecutor(max_workers=max(1, self.n_workers // 2)) as executor:
            sorted_sub_chunks = list(executor.map(np.sort, sub_chunks))

        return self._merge_sorted_arrays(sorted_sub_chunks)

    async def _multi_pass_external_sort(self, arr: npt.NDArray) -> npt.NDArray:
        file_path = "temp_external_sort.bin"
        dtype = arr.dtype
        arr.tofile(file_path)
        memory_limit = psutil.virtual_memory().available * 0.5
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)

        num_chunks = math.ceil(len(arr) * arr.itemsize / chunk_size)

        if len(arr) * arr.itemsize < memory_limit:
            return await self._external_sort(arr)

        sorted_chunks = []

        async def process_chunk(i):
            offset = i * chunk_size
            size = min(chunk_size, len(arr) * arr.itemsize - offset)
            chunk = await self._read_chunk(file_path, offset, size, dtype)
            return np.sort(chunk)

        for pass_num in tqdm(range(10), desc="Multi-pass external sort", leave=False):
            tasks = []
            for i in range(num_chunks):
                if (pass_num == 0 or i % (pass_num * 2) == 0) and (i + pass_num * 2 < num_chunks or pass_num == 0):
                    tasks.append(process_chunk(i))

            chunks = await asyncio.gather(*tasks)
            if not chunks:
                break

            sorted_chunks = self._merge_sorted_arrays(sorted_chunks + list(chunks))

            num_chunks = math.ceil(len(sorted_chunks) * arr.itemsize / chunk_size)

        os.remove(file_path)
        return sorted_chunks

    async def _incremental_sort(self, data_stream: Generator) -> npt.NDArray:
        file_path = "temp_incremental_sort.bin"
        temp_arrays = []

        for i, chunk in enumerate(tqdm(data_stream, desc="Processing stream", leave=False)):
            chunk_array = np.array(chunk)
            sorted_chunk = np.sort(chunk_array)
            temp_arrays.append(sorted_chunk)
            sorted_arr = self._merge_sorted_arrays(temp_arrays)
            sorted_arr.tofile(file_path)

            if i > 10:
                break

        with open(file_path, 'rb') as f:
            sorted_arr = np.fromfile(f, dtype=np.float64)
        os.remove(file_path)
        return sorted_arr

    def _hot_swap_sort(self, arr: npt.NDArray) -> Tuple[npt.NDArray, SortStats]:
        if len(arr) < 1000:
            return np.sort(arr), self._calculate_stats(arr, "hot_swap", Algorithm.INSERTIONSORT.value)

        if self.strategy == SortStrategy.ADAPTIVE:
            if np.std(arr) < (np.max(arr) - np.min(arr)) / 100:
                self.strategy = SortStrategy.HYBRID
                return self._hybrid_sort(arr), self._calculate_stats(arr, "hot_swap", Algorithm.MERGESORT.value)
            else:
                return np.sort(arr), self._calculate_stats(arr, "hot_swap", Algorithm.TIMSORT.value)
        elif self.strategy == SortStrategy.HYBRID:
            return self._advanced_block_sort(arr), self._calculate_stats(arr, "hot_swap", Algorithm.INTROSORT.value)
        else:
            return self._parallel_sort(arr)

    def _performance_dashboard(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
        processed_items = self.metrics.metrics
        current_strategy = self.strategy
        return {
            "cpu_usage": f"{cpu_percent:.1f}%",
            "memory_usage": f"{mem_usage:.1f}%",
            "processed_items": processed_items,
            "current_strategy": current_strategy
        }

    def measure_sort_performance(arr: npt.NDArray, strategy: SortStrategy) -> float:
        start_time = time.perf_counter()
        sorted_arr, _ = asyncio.run(self._sort_with_fallback(arr, strategy))
        end_time = time.perf_counter()
        return end_time - start_time

    def choose_best_strategy(arr: npt.NDArray) -> SortStrategy:
        strategies = [SortStrategy.BUCKET_SORT, SortStrategy.INTROSORT, SortStrategy.TIMSORT]
        best_strategy = strategies[0]
        best_time = float('inf')
        for strategy in strategies:
            time_taken = measure_sort_performance(arr, strategy)
            if time_taken < best_time:
                best_time = time_taken
                best_strategy = strategy
        return best_strategy
    
    def _predictive_analytics(self, arr: npt.NDArray) -> dict:

        n = len(arr)
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.data_type != "number":
            is_nearly_sorted = False
        else:
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1

        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        data_skewness = skew(sample)
        data_kurtosis = kurtosis(sample)

        features = np.array([std_dev, range_size, is_nearly_sorted, len(
            arr), data_skewness, data_kurtosis]).reshape(1, -1)

        if self.ml_model:
            predicted_strategy = np.argmax(
                self.ml_model.predict(features, verbose=0)[0])
        else:
            predicted_strategy = 0

        if predicted_strategy == 0:
            strategy_name = SortStrategy.ADAPTIVE
        elif predicted_strategy == 1:
            strategy_name = SortStrategy.BUCKET_SORT
        elif predicted_strategy == 2:
            strategy_name = SortStrategy.HYBRID
        elif predicted_strategy == 3:
            strategy_name = SortStrategy.PARALLEL
        elif predicted_strategy == 4:
            strategy_name = SortStrategy.RADIX_SORT
        elif predicted_strategy == 5:
            strategy_name = SortStrategy.COMPRESSION_SORT
        else:
            strategy_name = SortStrategy.ADAPTIVE

        estimated_time = 0.00001 * n + 0.000001 * n**2
        estimated_memory = n * arr.itemsize

        return {
            "estimated_time": f"{estimated_time:.4f}s",
            "estimated_memory": f"{estimated_memory / (1024 * 1024):.2f} MB",
            "predicted_strategy": strategy_name.value
        }

    def _predictive_feedback_loop(self, arr: npt.NDArray, strategy: SortStrategy, stats: SortStats):
        if not self.ml_model:
            return

        n = len(arr)
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.data_type != "number":
            is_nearly_sorted = False
        else:
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1
        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        data_skewness = skew(sample)
        data_kurtosis = kurtosis(sample)

        features = np.array([std_dev, range_size, is_nearly_sorted, len(
            arr), data_skewness, data_kurtosis]).reshape(1, -1)

        predicted_strategy = np.argmax(
            self.ml_model.predict(features, verbose=0)[0])
        if predicted_strategy != strategy.value:
            self.logger.info(
                f"Model mispredicted strategy: {strategy}, actual time {stats.execution_time}")

    def _real_time_predictor(self, arr: npt.NDArray) -> Dict[str, Any]:
        start_time = time.perf_counter()
        sample_size = min(1000, len(arr))
        sample = arr[np.random.choice(len(arr), sample_size, replace=False)]

        if self.use_ml_prediction:
            predicted_strategy = self._predict_strategy(sample)
        else:
            predicted_strategy = self._choose_optimal_strategy(sample)

        end_time = time.perf_counter()

        time_per_item = (end_time - start_time) / \
            sample_size if sample_size > 0 else 0
        estimated_time = time_per_item * len(arr)

        estimated_memory = len(arr) * arr.itemsize

        return {
            "estimated_time": f"{estimated_time:.4f}s",
            "estimated_memory": f"{estimated_memory / (1024 * 1024):.2f} MB",
            "suggested_strategy": predicted_strategy
        }

    def _benchmark_on_the_fly(self, arr: npt.NDArray, algorithm: Algorithm) -> None:
        start_time = time.perf_counter()
        if algorithm == Algorithm.QUICKSORT:
            self._quicksort(arr)
        elif algorithm == Algorithm.MERGESORT:
            np.sort(arr)
        elif algorithm == Algorithm.HEAPSORT:
            self._heapsort(arr)
        elif algorithm == Algorithm.TIMSORT:
            np.sort(arr, kind='stable')
        elif algorithm == Algorithm.INTROSORT:
            self._introsort(arr)
        elif algorithm == Algorithm.RADIXSORT:
            self._radix_sort(arr)
        elif algorithm == Algorithm.EXTERNALMERGESORT:
            asyncio.run(self._external_sort(arr))
        elif algorithm == Algorithm.COUNTINGSORT:
            self._counting_sort(arr)

        end_time = time.perf_counter()
        self.metrics.record(f'benchmark_{algorithm.value}_time', end_time - start_time)

    def _streaming_hybrid_sort(self, data_stream: Generator) -> Tuple[npt.NDArray, SortStats]:
        chunk_size = self._adaptive_chunk_size(1000, 8)

        chunks = []
        for chunk in tqdm(data_stream, desc="Processing stream", leave=False):
            chunk_array = np.array(chunk)
            if len(chunk_array) < chunk_size:
                chunks.append(np.sort(chunk_array))
            else:
                sub_chunks = np.array_split(chunk_array, max(1, len(chunk_array) // chunk_size))
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    sorted_chunks = list(tqdm(executor.map(np.sort, sub_chunks), total=len(sub_chunks), desc="Sorting sub-chunks", leave=False))
                chunks.extend(sorted_chunks)

        if not chunks:
            return np.array([]), self._calculate_stats(np.array([]), "stream_hybrid", "none", stream_chunks=0)

        merged_array = self._merge_sorted_arrays(chunks)
        return merged_array, self._calculate_stats(merged_array, "stream_hybrid", Algorithm.MERGESORT.value, stream_chunks=len(chunks))

    async def _sort_with_fallback(self, arr: npt.NDArray, strategy: SortStrategy) -> Tuple[npt.NDArray, SortStats]:
        try:
            if self.load_balancer_enabled:
                self._dynamic_cpu_load_balancing(arr)

            if self.distributed:
                if strategy == SortStrategy.EXTERNAL_SORT:
                    result = await self._external_sort(arr)
                    stats = self._calculate_stats(
                        arr, "external", Algorithm.EXTERNALMERGESORT.value)
                    result = (result, stats)
                else:
                    from HyperionSort import create_sort_handler
                    sorter = create_sort_handler(strategy=strategy.value, n_workers=self.n_workers, chunk_size=self.chunk_size,
                                                 cache_size=self.cache.size, adaptive_threshold=self.adaptive_threshold, block_size=self.block_manager.block_size)
                    result = await sorter.sort(arr)
            else:
                if self.load_balancer_enabled:
                    self._dynamic_cpu_load_balancing(arr)

                if strategy == SortStrategy.MEMORY_EFFICIENT:
                    result = self._memory_efficient_sort(arr)
                    gc.collect()

                elif strategy == SortStrategy.HYBRID:
                    result = (self._hybrid_sort(arr),
                              self._calculate_stats(arr, "hybrid", Algorithm.INTROSORT.value))
                elif strategy == SortStrategy.ADAPTIVE:
                    result = self._adaptive_sort(arr)
                elif strategy == SortStrategy.STREAM:
                    result = self._stream_sort_and_collect(arr)
                elif strategy == SortStrategy.BLOCK_SORT:
                    result = self._advanced_block_sort(arr)
                elif strategy == SortStrategy.BUCKET_SORT:
                    result = self._bucket_sort(arr)
                elif strategy == SortStrategy.RADIX_SORT:
                    result = (self._radix_sort(arr), self._calculate_stats(
                        arr, "radix", Algorithm.RADIXSORT.value))
                elif strategy == SortStrategy.COMPRESSION_SORT:
                    result, compression_ratio = self._pre_sort_compression(arr)
                    stats = self._calculate_stats(
                        arr, "compression", Algorithm.MERGESORT.value, compression_ratio=compression_ratio)
                    result = (result, stats)
                elif strategy == SortStrategy.EXTERNAL_SORT:
                    result = await self._external_sort(arr)
                    stats = self._calculate_stats(
                        arr, "external", Algorithm.EXTERNALMERGESORT.value)
                    result = (result, stats)
                elif strategy == SortStrategy.COUNTING_SORT:
                    result = (self._counting_sort(arr), self._calculate_stats(
                        arr, "counting", Algorithm.COUNTINGSORT.value))
                    gc.collect()

                elif strategy == SortStrategy.LAZY_SORT:
                    result = (self._lazy_sort(arr, len(
                        arr) // 2), self._calculate_stats(arr, "lazy", Algorithm.QUICKSELECT.value))
                elif strategy == SortStrategy.SEQUENTIAL_SORT:
                    result = (self._sequential_smart_sort(arr), self._calculate_stats(
                        arr, "sequential", Algorithm.TIMSORT.value))
                elif strategy == SortStrategy.MICRO_SORT:
                    result = (self._micro_sort(arr), self._calculate_stats(
                        arr, "micro", Algorithm.INSERTIONSORT.value))
                elif strategy == SortStrategy.HYBRID_COMPRESSION_SORT:
                    result, compression_ratio = self._hybrid_compression_sort(
                        arr)
                    stats = self._calculate_stats(
                        arr, "hybrid_compression", Algorithm.MERGESORT.value, compression_ratio=compression_ratio)
                    result = (result, stats)
                elif strategy == SortStrategy.HOT_SWAP_SORT:
                    result = self._hot_swap_sort(arr)
                    if not isinstance(result, tuple):
                        result = (result, self._calculate_stats(
                            arr, "hot_swap", Algorithm.INSERTIONSORT.value))
                else:
                    result = self._parallel_sort(arr)

            if isinstance(result, tuple):
                sorted_arr = result[0]
            else:
                sorted_arr = result

            if self.data_type == "number":
                is_sorted = np.all(sorted_arr[:-1] <= sorted_arr[1:])
            else:
                is_sorted = True
                for i in range(len(sorted_arr)-1):
                    if self._dynamic_comparator(sorted_arr[i], sorted_arr[i+1]) > 0:
                        is_sorted = False
                        break

            if not is_sorted:
                self.logger.warning(
                    f"Sort verification failed, switching to fallback strategy: {self.fallback_strategy}")
                fallback_result, fallback_stats = self._fallback_strategy(
                    arr, strategy)
                return fallback_result, fallback_stats
            return result
        except Exception as e:
            self.logger.error(f"Error during sorting: {e}", exc_info=True)
            return arr, self._calculate_stats(arr, "failed", "none", error=e)

    async def sort(
            self,
            arr: Union[List[Any], npt.NDArray, Generator],
            weights: Optional[npt.NDArray] = None,
            accuracy: Optional[float] = None,
            k: Optional[int] = None,
            top: Optional[bool] = False
        ) -> Tuple[npt.NDArray, SortStats]:
            self.start_time = time.perf_counter()
            self.logger.info("Starting sort operation...")

            if isinstance(arr, list):
                arr = self._process_mixed_data(arr)
                if not arr.size:
                    return np.array([]), self._calculate_stats(np.array([]), "failed", "none", error_detected=True)

            original_arr = np.array(arr, copy=True)
            if isinstance(arr, Generator) or self.stream_mode:
                if self.strategy == SortStrategy.STREAM:
                    return self._stream_sort_and_collect(arr)
                elif self.strategy == SortStrategy.STREAMING_HYBRID_SORT:
                    return self._streaming_hybrid_sort(arr)
                else:
                    return self._incremental_sort(arr)
            if self.profile:
                if not hasattr(self, 'profiler') or self.profiler is None:
                    self.profiler = cProfile.Profile()
                    self.profiler.enable()

            try:
                if not self._validate_data(arr):
                    return original_arr, self._calculate_stats(original_arr, "failed", "none", error_detected=True)

                if self.deduplicate_sort:
                    arr = self._deduplicate(arr)

                if self.distributed:
                    from HyperionSort import create_sort_handler
                    sorter = create_sort_handler(strategy=self.strategy.value, external_sort_threshold=self.external_sort_threshold, n_workers=self.n_workers, chunk_size=self.chunk_size, cache_size=self.cache.size,
                                                adaptive_threshold=self.adaptive_threshold, block_size=self.block_manager.block_size, use_ml_prediction=self.use_ml_prediction, data_type=self.data_type, log_level=self.logger.level)
                    return await sorter.sort(arr, k, top)
                else:
                    self._intelligent_parallelization(arr)
                    if weights is not None:
                        arr = self._priority_sort(arr, weights)
                    if accuracy is not None:
                        arr = self._approximate_sort(arr, accuracy)
                    strategy = self._dry_run_performance_check(arr)
                    self._dynamic_performance_monitoring(arr)
                    self._real_time_forecasting(arr)

                    if len(arr) > self.external_sort_threshold:
                        self.logger.info(
                            f"Using external sort for {len(arr):,} elements.")
                        strategy = SortStrategy.EXTERNAL_SORT
                    elif self.use_ml_prediction:
                        strategy = self._predict_strategy(arr)
                    elif self.strategy == SortStrategy.AUTO:
                        strategy = self._choose_optimal_strategy(arr)
                    else:
                        strategy = self.strategy

                    self.logger.info(f"Selected strategy: {strategy.value}")
                    key = self._create_historical_key(arr)
                    cached_strategy = self._get_cached_strategy(key)

                    if cached_strategy:
                        self.logger.info(
                            f"Applying cached strategy {cached_strategy.value} for the dataset.")
                        strategy = cached_strategy

                    if strategy == SortStrategy.LAZY_SORT:
                        if k is None:
                            k = len(arr) // 2
                        result = self._lazy_sort(arr, k, top)
                        stats = self._calculate_stats(result, "lazy", Algorithm.QUICKSELECT.value)
                    elif strategy == SortStrategy.MICRO_SORT and len(arr) > 1000:
                        result, stats = await self._sort_with_fallback(arr, strategy)
                    elif strategy == SortStrategy.HOT_SWAP_SORT and len(arr) > 1000:
                        result = self._hot_swap_sort(arr)
                        if not isinstance(result, tuple):
                            result = (result, self._calculate_stats(
                                arr, "hot_swap", Algorithm.INSERTIONSORT.value))
                        stats = result[1]
                    else:
                        result, stats = await self._sort_with_fallback(arr, strategy)

                if self.profile:
                    self.profiler.disable()
                    s = io.StringIO()
                    ps = pstats.Stats(
                        self.profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats()
                    self.logger.debug(f"Profile results:\n{s.getvalue()}")

                if isinstance(result, tuple):
                    self._predictive_feedback_loop(arr, strategy, result[1])
                    self._cache_historical_runs(key, strategy)
                    self._save_run_history(arr, strategy, result[1])
                    self.logger.info("Sort operation completed successfully")
                    return result

                self._cache_historical_runs(key, strategy)
                self._save_run_history(arr, strategy, stats)
                self.logger.info("Sort operation completed successfully")
                return result, stats

            except Exception as e:
                self.logger.error(f"Error during sorting: {e}", exc_info=True)
                return original_arr, self._calculate_stats(original_arr, "failed", "none", error=e)
        
    def _setup_logging(self, level: int):
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handler.setLevel(level)
        self.logger.addHandler(handler)

    @contextmanager
    def _resource_monitor(self):
        start_mem = psutil.Process().memory_info().rss
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_mem = psutil.Process().memory_info().rss

            self.metrics.record('memory_delta', end_mem - start_mem)
            self.metrics.record('operation_time', end_time - start_time)
            if self.eco_mode:
                self._dynamic_logging(logging.INFO)

    def _adaptive_chunk_size(self, arr_size, itemsize):
        available_memory = psutil.virtual_memory().available
        total_size = arr_size * itemsize
        cpu_count = psutil.cpu_count()

        l3_cache = psutil.cpu_count() * 2 ** 20

        if total_size < l3_cache:
            return min(arr_size, 10000)

        optimal_chunks = max(
            cpu_count,
            int(total_size / (available_memory * 0.8))
        )

        return max(1000, arr_size // optimal_chunks)

    def _advanced_partition(self, arr: npt.NDArray) -> List[npt.NDArray]:
        if len(arr) < 1000:
            return [arr]

        quantiles = np.linspace(0, 100, min(len(arr) // 1000 + 1, 10))
        pivots = np.percentile(arr, quantiles)

        partitions = []
        start_idx = 0

        for i in range(1, len(pivots)):
            mask = (arr >= pivots[i - 1]) & (arr < pivots[i])
            partition = arr[mask]
            if len(partition) > 0:
                partitions.append(partition)

        return partitions

    def hybrid_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)
        elif len(arr) < 10000:
            return self._introsort(arr)
        else:
            return np.sort(arr, kind='stable')

    def dynamic_sampling(self, arr: npt.NDArray) -> npt.NDArray:
        sample_size = min(1000, len(arr))
        sample = arr[np.random.choice(len(arr), sample_size, replace=False)]
        return sample

    def compression_aware_sort(self, arr: npt.NDArray) -> npt.NDArray:
        compressed_data = compress(arr.tobytes())
        decompressed_arr = np.frombuffer(decompress(compressed_data), dtype=arr.dtype)
        return np.sort(decompressed_arr)

    def early_exit_optimization(self, arr: npt.NDArray) -> npt.NDArray:
        if np.all(arr[:-1] <= arr[1:]):
            return arr
        return np.sort(arr)

    def _merge_sorted_arrays(self, arrays: List[np.ndarray]) -> np.ndarray:
        merged = []
        heap = []

        for i, arr in enumerate(arrays):
            if len(arr) > 0:
                heapq.heappush(heap, (arr[0], i, 0))

        while heap:
            val, arr_idx, elem_idx = heapq.heappop(heap)
            merged.append(val)

            if elem_idx + 1 < len(arrays[arr_idx]):
                next_val = arrays[arr_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))

        return np.array(merged)

    def _adaptive_sort(self, arr: npt.NDArray) -> Tuple[npt.NDArray, SortStats]:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        n = len(arr)
        if n <= 1:
            return arr, self._calculate_stats(arr, "adaptive", Algorithm.TIMSORT.value)

        try:
            sample_size = min(1000, n)
            sample_indices = np.random.choice(n, sample_size, replace=False)
            sample = arr[sample_indices]

            std_dev = np.std(sample)
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1

            if is_nearly_sorted:
                algorithm = Algorithm.TIMSORT
                with self._resource_monitor():
                    result = np.sort(arr, kind='stable')
            elif std_dev < (np.max(sample) - np.min(sample)) / 100:
                algorithm = Algorithm.QUICKSORT
                with self._resource_monitor():
                    result = self._parallel_sort(arr)[0]
            else:
                algorithm = Algorithm.MERGESORT
                with self._resource_monitor():
                    result = self._hybrid_sort(arr)

            return result, self._calculate_stats(arr, "adaptive", algorithm.value)

        except Exception as e:
            with self._resource_monitor():
                result = np.sort(arr)
            return result, self._calculate_stats(arr, "adaptive", Algorithm.TIMSORT.value)

    def _optimize_bucket_count(self, arr: npt.NDArray) -> int:
        n = len(arr)
        if n < 1000:
            return max(1, n // 10)

        cache_key = f"bucket_count_{n}_{arr.std():.2f}"
        if cached_value := self.cache.get(cache_key):
            return cached_value

        if self.data_type == "number":
            sample_size = min(1000, n)
            sample = arr[np.random.choice(n, sample_size, replace=False)]
            std_dev = np.std(sample)
            range_size = np.ptp(sample)
        else:
            std_dev = 1
            range_size = 1

        if std_dev < range_size / 100:
            bucket_count = int(math.sqrt(n))
        else:
            density = n / range_size if range_size > 0 else 1
            bucket_count = int(min(math.sqrt(n) * (std_dev / range_size) * 2 if range_size > 0 else math.sqrt(n), n / math.log2(n)))

        if np.isnan(bucket_count) or bucket_count <= 0:
            bucket_count = max(1, n // 10)

        self.cache.put(cache_key, bucket_count)
        return bucket_count

    def _calculate_stats(self, arr: npt.NDArray, strategy: str, algorithm: str, error: Optional[Exception] = None, stream_chunks: int = 0, compression_ratio: float = 1.0, error_detected: bool = False) -> SortStats:
        process = psutil.Process()
        disk_io_end = psutil.disk_io_counters()
        disk_read_count = disk_io_end.read_bytes - self.metrics.disk_io_start.read_bytes
        disk_write_count = disk_io_end.write_bytes - \
            self.metrics.disk_io_start.write_bytes
        performance = PerformanceMetrics(
            cpu_time=time.process_time(),
            wall_time=time.perf_counter() - self.start_time,
            memory_peak=process.memory_info().rss / (1024 * 1024),
            cache_hits=self.cache.hits,
            cache_misses=self.cache.misses,
            thread_count=len(process.threads()),
            context_switches=process.num_ctx_switches().voluntary,
            disk_io=disk_read_count + disk_write_count,
            compression_ratio=compression_ratio
        )

        if error:
            return SortStats(
                execution_time=time.perf_counter() - self.start_time,
                memory_usage=performance.memory_peak,
                items_processed=len(arr),
                cpu_usage=psutil.cpu_percent(),
                bucket_distribution=[],
                strategy_used=strategy,
                algorithm_used="none",
                performance=performance,
                optimization_history=self.metrics.metrics,
                stream_chunks=stream_chunks,
                compression_ratio=compression_ratio,
                error_detected=True,
                fallback_strategy=self.fallback_strategy.value
            )

        return SortStats(
            execution_time=performance.wall_time,
            memory_usage=performance.memory_peak,
            items_processed=len(arr),
            cpu_usage=psutil.cpu_percent(),
            bucket_distribution=self._get_bucket_distribution(arr),
            strategy_used=strategy,
            algorithm_used=algorithm,
            performance=performance,
            optimization_history=self.metrics.metrics,
            stream_chunks=stream_chunks,
            compression_ratio=compression_ratio,
            error_detected=error_detected,
            fallback_strategy=self.fallback_strategy.value
        )

    def _choose_optimal_strategy(self, arr: npt.NDArray) -> SortStrategy:
        n = len(arr)

        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.data_type != "number":
            is_nearly_sorted = False
        else:
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1

        if self.data_type != "number":
            std_dev = 1
            range_size = 1
        else:
            std_dev = np.std(sample)
            range_size = np.ptp(sample)

        memory_available = psutil.virtual_memory().available
        if self.data_type == "number":
            data_skewness = skew(sample)
            data_kurtosis = kurtosis(sample)
        else:
            data_skewness = 0
            data_kurtosis = 0

        estimated_memory = n * arr.itemsize * 3

        if estimated_memory > memory_available * 0.7:
            return SortStrategy.MEMORY_EFFICIENT

        if data_skewness > 2 or data_kurtosis > 5:
            return SortStrategy.BUCKET_SORT

        if is_nearly_sorted:
            return SortStrategy.ADAPTIVE

        if n > 1_000_000 and psutil.cpu_count() > 2:
            return SortStrategy.PARALLEL

        if std_dev < range_size / 100:
            return SortStrategy.HYBRID

        if n > self.external_sort_threshold:
            return SortStrategy.EXTERNAL_SORT

        duplicate_ratio = self._analyze_duplicate_ratio(arr)
        if duplicate_ratio > self.duplicate_threshold:
            return SortStrategy.COUNTING_SORT

        if n > 10000:
            return SortStrategy.SEQUENTIAL_SORT

        if n > 1000:
            return SortStrategy.MICRO_SORT

        return SortStrategy.ADAPTIVE

    def _get_bucket_distribution(self, arr: npt.NDArray) -> List[int]:
        if len(arr) == 0:
            return []
        if self.data_type != "number":
            return []

        n_buckets = self._optimize_bucket_count(arr)
        if n_buckets == 0:
            return []
        bucket_ranges = np.linspace(arr.min(), arr.max(), n_buckets + 1)
        distribution = []

        for i in range(n_buckets):
            mask = (arr >= bucket_ranges[i]) & (arr < bucket_ranges[i + 1])
            distribution.append(int(np.sum(mask)))

        return distribution

    def _shell_sort(self, arr: npt.NDArray) -> npt.NDArray:
        n = len(arr)
        gap = n // 2

        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 2

        return arr

    def _comb_sort(self, arr: npt.NDArray) -> npt.NDArray:
        n = len(arr)
        gap = n
        shrink = 1.3
        sorted = False

        while not sorted:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted = True

            for i in range(n - gap):
                if arr[i] > arr[i + gap]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                    sorted = False

        return arr

    def _pancake_sort(self, arr: npt.NDArray) -> npt.NDArray:
        def flip(arr, i):
            start = 0
            while start < i:
                arr[start], arr[i] = arr[i], arr[start]
                start += 1
                i -= 1

        n = len(arr)
        for curr_size in range(n, 1, -1):
            mi = np.argmax(arr[:curr_size])
            if mi != curr_size - 1:
                flip(arr, mi)
                flip(arr, curr_size - 1)

        return arr

    def _gnome_sort(self, arr: npt.NDArray) -> npt.NDArray:
        n = len(arr)
        index = 0

        while index < n:
            if index == 0 or arr[index] >= arr[index - 1]:
                index += 1
            else:
                arr[index], arr[index - 1] = arr[index - 1], arr[index]
                index -= 1

        return arr

    def _cycle_sort(self, arr: npt.NDArray) -> npt.NDArray:
        n = len(arr)
        writes = 0

        for cycle_start in range(0, n - 1):
            item = arr[cycle_start]
            pos = cycle_start

            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1

            if pos == cycle_start:
                continue

            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            writes += 1

            while pos != cycle_start:
                pos = cycle_start
                for i in range(cycle_start + 1, n):
                    if arr[i] < item:
                        pos += 1

                while item == arr[pos]:
                    pos += 1
                arr[pos], item = item, arr[pos]
                writes += 1

        return arr

    def _bitonic_sort(self, arr: npt.NDArray) -> npt.NDArray:
        def bitonic_merge(arr, low, cnt, direction):
            if cnt > 1:
                k = cnt // 2
                for i in range(low, low + k):
                    if (direction == 1 and arr[i] > arr[i + k]) or (direction == 0 and arr[i] < arr[i + k]):
                        arr[i], arr[i + k] = arr[i + k], arr[i]
                bitonic_merge(arr, low, k, direction)
                bitonic_merge(arr, low + k, k, direction)

        def bitonic_sort(arr, low, cnt, direction):
            if cnt > 1:
                k = cnt // 2
                bitonic_sort(arr, low, k, 1)
                bitonic_sort(arr, low + k, k, 0)
                bitonic_merge(arr, low, cnt, direction)

        n = len(arr)
        bitonic_sort(arr, 0, n, 1)
        return arr

    def _odd_even_sort(self, arr: npt.NDArray) -> npt.NDArray:
        n = len(arr)
        sorted = False

        while not sorted:
            sorted = True
            for i in range(1, n - 1, 2):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    sorted = False

            for i in range(0, n - 1, 2):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    sorted = False

        return arr

    def _stooge_sort(self, arr: npt.NDArray, l: int, h: int) -> npt.NDArray:
        if l >= h:
            return arr

        if arr[l] > arr[h]:
            arr[l], arr[h] = arr[h], arr[l]

        if h - l + 1 > 2:
            t = (h - l + 1) // 3
            self._stooge_sort(arr, l, h - t)
            self._stooge_sort(arr, l + t, h)
            self._stooge_sort(arr, l, h - t)

        return arr

    def _smooth_sort(self, arr: npt.NDArray) -> npt.NDArray:
        def sift(arr, pshift, head):
            while pshift > 1:
                rt = head - 1
                lf = head - 1 - (1 << (pshift - 2))
                if arr[head] >= arr[lf] and arr[head] >= arr[rt]:
                    break
                if arr[lf] >= arr[rt]:
                    arr[head], arr[lf] = arr[lf], arr[head]
                    head = lf
                    pshift -= 1
                else:
                    arr[head], arr[rt] = arr[rt], arr[head]
                    head = rt
                    pshift -= 2

        def trinkle(arr, p, pshift, head, trusty):
            while p != 1:
                stepson = head - pshift
                if arr[stepson] <= arr[head]:
                    break
                if not trusty and pshift > 1:
                    rt = head - 1
                    lf = head - 1 - (1 << (pshift - 2))
                    if arr[rt] >= arr[stepson] or arr[lf] >= arr[stepson]:
                        break
                arr[head], arr[stepson] = arr[stepson], arr[head]
                head = stepson
                trail = p & -p
                p >>= trail.bit_length()
                pshift += trail.bit_length()

            sift(arr, pshift, head)

        def smooth_sort(arr):
            n = len(arr)
            q = 1
            p = 1
            pshift = 1
            head = 0

            while q < n:
                if (p & 3) == 3:
                    sift(arr, pshift, head)
                    p >>= 2
                    pshift += 2
                else:
                    if (p & 1) == 1:
                        sift(arr, pshift, head)
                    else:
                        trinkle(arr, p, pshift, head, False)
                    if pshift == 1:
                        p <<= 1
                        pshift -= 1
                    else:
                        p <<= pshift - 1
                        pshift = 1
                p += 1
                q += 1
                head += 1

            trinkle(arr, p, pshift, head, False)

            while q > 1:
                q -= 1
                head -= 1
                if pshift > 1:
                    p -= 1
                    pshift -= 1
                    trinkle(arr, p, pshift, head, True)
                    p <<= 1
                    pshift += 1
                    trinkle(arr, p, pshift, head, True)
                else:
                    p >>= 1
                    pshift += 1

        smooth_sort(arr)
        return arr

    def _priority_sort(self, arr: npt.NDArray, weights: npt.NDArray) -> npt.NDArray:
        if len(arr) != len(weights):
            raise ValueError("Array and weights must be of the same length")

        weighted_arr = [(arr[i], weights[i]) for i in range(len(arr))]
        weighted_arr.sort(key=lambda x: x[1], reverse=True)
        sorted_arr = np.array([x[0] for x in weighted_arr])
        return sorted_arr
    
    def _weighted_bucket_sort(self, arr: npt.NDArray, weights: npt.NDArray) -> npt.NDArray:
        if len(arr) != len(weights):
            raise ValueError("Array and weights must be of the same length")

        n_buckets = self._optimize_bucket_count(arr)
        min_val, max_val = arr.min(), arr.max()

        if min_val == max_val:
            return arr

        buckets = [[] for _ in range(n_buckets)]
        width = (max_val - min_val) / n_buckets

        for i in range(len(arr)):
            bucket_idx = int((arr[i] - min_val) / width)
            if bucket_idx == n_buckets:
                bucket_idx -= 1
            buckets[bucket_idx].append((arr[i], weights[i]))

        sorted_buckets = []
        for bucket in buckets:
            bucket.sort(key=lambda x: x[1], reverse=True)
            sorted_buckets.extend([x[0] for x in bucket])

        return np.array(sorted_buckets)
    
    def _approximate_sort(self, arr: npt.NDArray, accuracy: float = 0.95) -> npt.NDArray:
        if not (0 < accuracy <= 1):
            raise ValueError("Accuracy must be between 0 and 1")

        k = int(len(arr) * accuracy)
        return self._lazy_sort(arr, k)
    
    def _parallel_sort(self, arr: npt.NDArray) -> Tuple[npt.NDArray, SortStats]:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        chunks = np.array_split(arr, max(1, len(arr) // chunk_size))
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_chunks = list(executor.map(np.sort, chunks))

        result = self._merge_sorted_arrays(sorted_chunks)
        return result, self._calculate_stats(arr, "parallel", Algorithm.QUICKSORT.value)

    def _memory_efficient_sort(self, arr: npt.NDArray) -> Tuple[npt.NDArray, SortStats]:
        chunk_size = self._adaptive_chunk_size(len(arr), arr.itemsize)
        result = np.zeros_like(arr)
        for i in range(0, len(arr), chunk_size):
            chunk = arr[i:i + chunk_size]
            np.sort(chunk)
            result[i:i + chunk_size] = chunk
        gc.collect()
        return result, self._calculate_stats(arr, "memory_efficient", Algorithm.MERGESORT.value)

    def _create_historical_key(self, arr: npt.NDArray) -> str:
        if self.data_type == "number":
            return f"{len(arr)}_{arr.dtype}_{arr.std():.2f}"
        else:
            return f"{len(arr)}_{arr.dtype}_{time.time()}_non_number"

    def _cache_historical_runs(self, key: str, strategy: SortStrategy):
        self.historical_runs[key] = strategy

    def _get_cached_strategy(self, key: str) -> Optional[SortStrategy]:
        return self.historical_runs.get(key)

    def _dynamic_logging(self, level: int = logging.INFO) -> None:
        if self.strategy != SortStrategy.AUTO:
            handler = self.logger.handlers[0]
            handler.setLevel(level)
            self.logger.info(f"Log level changed to {level} dynamically")

    def _counting_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0):
            self.logger.warning(
                "Counting Sort only works with non-negative integers.")
            return self._fallback_strategy(arr, SortStrategy.COUNTING_SORT)[0]

        min_val = np.min(arr)
        max_val = np.max(arr)

        if max_val > 1_000_000:
            self.logger.warning(
                "Data range for Counting Sort too large, using fallback.")
            return self._fallback_strategy(arr, SortStrategy.COUNTING_SORT)[0]

        counts = np.zeros(max_val - min_val + 1, dtype=int)

        for x in arr:
            counts[x-min_val] += 1

        sorted_arr = []
        for i, count in enumerate(counts):
            sorted_arr.extend([i+min_val] * count)
        return np.array(sorted_arr)

    def _extract_features(self, arr: npt.NDArray) -> dict:
        n = len(arr)
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]

        if self.data_type != "number":
            is_nearly_sorted = False
            
            if self.data_type == "string":
                lengths = np.array([len(str(x)) for x in sample])
                correlation = np.abs(np.corrcoef(lengths[:-1], lengths[1:])[0, 1])
            else:
                correlation = 0.0
            
            if self.data_type == "string":
                from collections import Counter
                char_counts = Counter("".join(sample))
                total_chars = sum(char_counts.values())
                probabilities = [count / total_chars for count in char_counts.values()]
                data_entropy = entropy(probabilities)
            else:
                data_entropy = 0.0
            
            if self.data_type == "string":
                unique_count = len(set(sample))
            else:
                unique_count = 0
                
        else:
            is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1
            correlation = np.abs(np.corrcoef(sample[:-1], sample[1:])[0, 1])
            
            hist, bin_edges = np.histogram(sample, bins='auto', density=True)
            data_entropy = entropy(hist)
            
            unique_count = len(np.unique(sample))

        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        data_skewness = skew(sample)
        data_kurtosis = kurtosis(sample)
        
        quantiles = np.percentile(sample, [25, 50, 75])

        return {
            "std_dev": std_dev,
            "range_size": range_size,
            "is_nearly_sorted": is_nearly_sorted,
            "n": n,
            "data_skewness": data_skewness,
            "data_kurtosis": data_kurtosis,
            "data_type": self.data_type,
            "itemsize": arr.itemsize,
            "correlation": correlation,
            "entropy": data_entropy,
            "unique_count": unique_count,
            "q25": quantiles[0],
            "q50": quantiles[1],
            "q75": quantiles[2]
        }

def benchmark(
    sorter: EnhancedHyperionSort,
    sizes: List[int],
    runs: int = 3,
    save_results: bool = True
) -> Dict[str, Any]:
    results = []
    training_data = []

    for size in sizes:
        size_results = []
        for run in range(runs):
            logger.info(
                f"\nBenchmarking với {size:,} phần tử (Run {run + 1}/{runs}):")
            if run == 0:
                data = np.random.randint(0, size * 10, size=size)
            elif run == 1:
                data = np.sort(np.random.randint(0, size * 10, size=size))
                data[::100] = np.random.randint(
                    0, size * 10, size=len(data[::100]))
            else:
                n_clusters = 10
                cluster_points = size // n_clusters
                clusters = []

                for i in range(n_clusters):
                    center = np.random.randint(0, size * 10)
                    cluster = np.random.normal(loc=center,
                                               scale=size / 100,
                                               size=cluster_points)
                    clusters.append(cluster)

                data = np.concatenate(clusters).astype(np.int32)
            features = sorter._extract_features(arr=data)

            sorted_arr, metrics = asyncio.run(sorter.sort(data))

            if isinstance(metrics, tuple):
                metrics = metrics[1]
                sorted_arr = sorted_arr[0]

            if sorter.data_type == "number":
                is_sorted = np.all(sorted_arr[:-1] <= sorted_arr[1:])
            else:
                is_sorted = True
                for i in range(len(sorted_arr)-1):
                    if sorter._dynamic_comparator(sorted_arr[i], sorted_arr[i+1]) > 0:
                        is_sorted = False
                        break

            print(f"\n✨ Kết quả chạy {run + 1}:")
            print(f"✓ Đã sort xong!")
            print(f"⚡ Thời gian thực thi: {metrics.execution_time:.4f} giây")
            print(f"⏱️ CPU time: {metrics.performance.cpu_time:.4f} giây")
            print(f"💾 Bộ nhớ sử dụng: {metrics.memory_usage:.2f} MB")
            print(f"🖥️ CPU usage: {metrics.cpu_usage:.1f}%")
            print(
                f"🚀 Tốc độ xử lý: {size/metrics.execution_time:,.0f} items/giây")
            print(f"🎯 Strategy: {metrics.strategy_used}")
            print(f"🔄 Algorithm: {metrics.algorithm_used}")
            print(
                f"💽 Disk IO: {metrics.performance.disk_io/ (1024 * 1024):.2f} MB")
            print(f"✓ Kết quả đúng: {is_sorted}")
            if metrics.error_detected:
                print(f"🚨 Fallback used: {metrics.fallback_strategy}")
            if metrics.compression_ratio < 1.0:
                print(f"🗜️ Compression ratio: {metrics.compression_ratio:.2f}")

            result_data = {
                'size': size,
                'run': run,
                'distribution': ['random', 'nearly_sorted', 'clustered'][run],
                'time': metrics.execution_time,
                'cpu_time': metrics.performance.cpu_time,
                'memory': metrics.memory_usage,
                'strategy': metrics.strategy_used,
                'algorithm': metrics.algorithm_used,
                'items_per_second': size / metrics.execution_time,
                'is_sorted': is_sorted,
                'compression_ratio': metrics.compression_ratio,
                'fallback_strategy': metrics.fallback_strategy,
                'disk_io': metrics.performance.disk_io/(1024*1024),
                **features
            }
            training_data.append(result_data)

            size_results.append(result_data)

        avg_time = np.mean([r['time'] for r in size_results])
        std_time = np.std([r['time'] for r in size_results])
        avg_disk_io = np.mean([r['disk_io'] for r in size_results])
        std_disk_io = np.std([r['disk_io'] for r in size_results])

        print(f"\n📊 Thống kê cho {size:,} phần tử:")
        print(f"📈 Thời gian trung bình: {avg_time:.4f} ± {std_time:.4f} giây")
        print(f"🎯 Độ ổn định: {(1 - std_time / avg_time) * 100:.2f}%")
        print(
            f"💾 Disk IO trung bình: {avg_disk_io:.2f} ± {std_disk_io:.2f} MB")

        results.extend(size_results)

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n💾 Đã lưu kết quả vào: {filename}")

        training_file = f"benchmark_train_{timestamp}.pkl"
        with open(training_file, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"\n💾 Training data is saved into: {training_file}")

    return {
        'results': results,
        'summary': {
            'total_runs': len(results),
            'sizes_tested': sizes,
            'best_performance': min(results, key=lambda x: x['time']),
            'worst_performance': max(results, key=lambda x: x['time'])
        },
        'training_data': training_data
    }


def create_sort_handler(strategy: str, **kwargs):
    from HyperionSort import EnhancedHyperionSort
    if is_distributed_env():
        logger.info("Detected distributed environment, using Dask/Ray")
        if ray.is_initialized():
            return EnhancedHyperionSort(strategy=SortStrategy(strategy), **kwargs, distributed=True)

        elif os.getenv('ENV_TYPE') == "dask":
            return EnhancedHyperionSort(strategy=SortStrategy(strategy), **kwargs,  distributed=True)
        else:
            return EnhancedHyperionSort(strategy=SortStrategy(strategy), **kwargs)

    else:
        logger.info(
            "Detected single machine environment, using enhanced hyperion sort.")
        return EnhancedHyperionSort(strategy=SortStrategy(strategy), **kwargs)


def create_test_training_data(training_sizes: List[int]) -> List[Dict[str, Any]]:
    test_training_data = []
    enhanced_sort_instance = EnhancedHyperionSort()

    for size in training_sizes:
        data = np.random.randint(0, size * 10, size=size)

        test_training_data.append(
            {
                'std_dev': np.std(data),
                'range_size': np.ptp(data),
                'is_nearly_sorted': np.sum(np.diff(data) < 0) < len(data) * 0.1,
                'n': len(data),
                 'data_skewness': np.std(data) / np.ptp(data) if np.ptp(data) > 0 else 0,
                'data_kurtosis': np.std(data) * 10 if np.ptp(data) > 0 else 0,
                   'data_type':"number",
                    'strategy': EnhancedHyperionSort.train_predict_label({
                        'std_dev': np.std(data),
                        'range_size': np.ptp(data), 
                        'is_nearly_sorted': np.sum(np.diff(data) < 0) < len(data) * 0.1,
                        'n': len(data),
                        'data_skewness': np.std(data) / np.ptp(data) if np.ptp(data) > 0 else 0,
                        'data_kurtosis': np.std(data) * 10 if np.ptp(data) > 0 else 0,
                        'data_type':'number'
                    })
            }
         )
    return test_training_data


def test_realtime_predict_logic(data_sizes: List[int], **kwargs) -> None:
    sorter = EnhancedHyperionSort(**kwargs)

    print("\n Real time predictor tests")
    for size in data_sizes:
        data = np.random.randint(0, size * 10, size=size)
        predictor = sorter._real_time_predictor(data)
        print(f"\nReal time predictor for data size: {size:,}")
        print(f"Estimated time: {predictor['estimated_time']}")
        print(f"Estimated memory: {predictor['estimated_memory']}")
        print(f"Suggested Strategy: {predictor['suggested_strategy'].value}")


if __name__ == "__main__":
    def test_critical_function(mocker):
        mocker.patch('HyperionSort.log_system_metrics')
        
        mock_logger = mocker.patch('HyperionSort.logger')
        
        critical_function()
        
        assert mock_logger.info.called
        assert "Sorting completed. Execution time:" in mock_logger.info.call_args[0][0]

    def test_sort_function():
        sorter = EnhancedHyperionSort()
        data = np.random.randint(0, 1000, size=1000)
        sorted_data, stats = asyncio.run(sorter.sort(data))
        
        assert np.all(sorted_data[:-1] <= sorted_data[1:])
        
        assert stats.items_processed == 1000
        assert stats.execution_time > 0

    def test_sort_strategy():
        sorter = EnhancedHyperionSort(strategy=SortStrategy.PARALLEL)
        data = np.random.randint(0, 1000, size=1000)
        sorted_data, stats = asyncio.run(sorter.sort(data))
        
        assert np.all(sorted_data[:-1] <= sorted_data[1:])
        
        assert stats.strategy_used == SortStrategy.PARALLEL.value

    profile_execution = True
    data_distribution_test = True
    benchmark_mode = True
    n_samples_train_for_test = 100_000
    training_sizes_for_tests = [100, 1000, 10_000, 100_000]

    enhanced_sort = EnhancedHyperionSort()
    test_training_data = create_test_training_data(training_sizes_for_tests)

    model_from_scratch = enhanced_sort._train_ml_models(n_samples_train=n_samples_train_for_test)
    model_with_test_data = enhanced_sort._train_ml_models(training_data_set=test_training_data, n_samples_train=n_samples_train_for_test)

    test_predict_data = np.array([
        [1, 50, 1, 100, 0, 0.5],
        [10, 100, 0, 100_000, 2.3, 6.2],
        [200, 100, 0, 1_000_000, 0.5, 1],
        [2, 40, 1, 100, -1, 0.4],
        [3, 5000, 1, 10_000, -1, -1],
        [40, 5, 0, 1_000, 0.1, 0.2],
        [4000, 100000, 1, 200000, 1.3, 2],
        [5, 5, 1, 100, -2, 0.1],
        [40, 1000, 0, 500, 0.3, 2],
    ], dtype=np.float64)

    for idx, pred_data in enumerate(test_predict_data):
        predicted_strategy = np.argmax(model_from_scratch[0].predict(np.array(pred_data).reshape(1, -1))[0])
        predicted_strategy_set = np.argmax(model_with_test_data[0].predict(np.array(pred_data).reshape(1, -1))[0])

        features_data = {
            'std_dev': pred_data[0],
            'range_size': pred_data[1],
            'is_nearly_sorted': pred_data[2],
            'n': pred_data[3],
            'data_skewness': pred_data[4],
            'data_kurtosis': pred_data[5],
            'data_type': 'number'
        }

        real_prediction = SortStrategy[enhanced_sort.train_predict_label(features_data)]

        try:
            predicted_strategy_name = SortStrategy(predicted_strategy).name
        except ValueError:
            predicted_strategy_name = "INVALID"

        try:
            predicted_strategy_set_name = SortStrategy(predicted_strategy_set).name
        except ValueError:
            predicted_strategy_set_name = "INVALID"

        print(f"Prediction Test Case {idx}: Raw data: {pred_data}")
        print(f"Random Generated Model predict {predicted_strategy_name} vs Dataset trained predict: {predicted_strategy_set_name}, Expected Result: {real_prediction.name} \n")
    
    if benchmark_mode:
        sorter = EnhancedHyperionSort(
            strategy=SortStrategy.AUTO,
            profile=profile_execution,
            cache_size=2000,
            adaptive_threshold=0.8,
            use_ml_prediction=True,
            external_sort_threshold=1000000,
            eco_mode=True,
            priority_mode="speed",
            deduplicate_sort=True,
            data_type="number",
            log_level=logging.INFO,
            benchmark=True,
            data_distribution_test=data_distribution_test
        )

        test_sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
        benchmark_results = benchmark(
            sorter=sorter,
            sizes=test_sizes,
            runs=3,
            save_results=True
        )

        plot_performance_metrics(benchmark_results['results'])
        plot_interactive_performance_metrics(benchmark_results['results'])
        analyze_performance_metrics(benchmark_results['results'])

        if data_distribution_test:
            def data_generator():
                for i in range(1000000):
                    yield np.random.randint(0, 1000)

            sorter = EnhancedHyperionSort(
                strategy=SortStrategy.STREAM,
                profile=False,
                cache_size=2000,
                adaptive_threshold=0.8,
                stream_mode=True,
                use_ml_prediction=False,
                data_type="number",
                log_level=logging.INFO
            )

            sorted_arr, stream_stats = asyncio.run(sorter.sort(data_generator()))
            print(f"Stream sort completed, total chunks: {stream_stats.stream_chunks}")
            print(f"Stream sort time : {stream_stats.execution_time:.4f} seconds")
            print(f"Stream sorted item: {len(sorted_arr):,} items")

            sorter = EnhancedHyperionSort(
                strategy=SortStrategy.LAZY_SORT,
                profile=False,
                cache_size=2000,
                adaptive_threshold=0.8,
                stream_mode=False,
                use_ml_prediction=False,
                data_type="number",
                log_level=logging.INFO
            )

            data_arr = np.random.randint(0, 1000, size=1_000_000)
            top_k = 100
            sorted_topk, lazy_stats = asyncio.run(sorter.sort(data_arr, k=top_k, top=True))
            print(f"\nLazy sort completed, top {top_k} elements")
            print(f"Lazy sort time : {lazy_stats.execution_time:.4f} seconds")
            print(f"Lazy sorted item: {len(sorted_topk):,} items")

            print("\n🌟 Kết quả tổng quan:")
            print(f"📊 Tổng số lần chạy: {benchmark_results['summary']['total_runs']}")
            print("\n🏆 Hiệu suất tốt nhất:")
            best = benchmark_results['summary']['best_performance']
            print(f"- Kích thước: {best['size']:,}")
            print(f"- Thời gian: {best['time']:.4f} giây")
            print(f"- Strategy: {best['strategy']}")
            print(f"- Distribution: {best['distribution']}")

            sorter = EnhancedHyperionSort(
                strategy=SortStrategy.AUTO,
                profile=False,
                cache_size=2000,
                adaptive_threshold=0.8,
                use_ml_prediction=False,
                external_sort_threshold=1000000,
                eco_mode=True,
                priority_mode="speed",
                deduplicate_sort=True,
                data_type="string",
                log_level=logging.INFO
            )

            string_data = ["apple", "banana", "cherry", "date", "fig", "apple", "banana", "date"]
            sorted_string, string_stats = asyncio.run(sorter.sort(string_data))
            print(f"\nString sort completed, time: {string_stats.execution_time:.4f} seconds")
            print(f"Sorted string result {sorted_string}")

            sorter = EnhancedHyperionSort(
                strategy=SortStrategy.AUTO,
                profile=False,
                cache_size=2000,
                adaptive_threshold=0.8,
                use_ml_prediction=False,
                external_sort_threshold=1000000,
                eco_mode=True,
                priority_mode="speed",
                deduplicate_sort=True,
                data_type="object",
                log_level=logging.INFO
            )

            object_data = [{"name": "bob", "age": 30}, {"name": "alice", "age": 25}, {"name": "bob", "age": 20}]
            sorted_object, object_stats = asyncio.run(sorter.sort(object_data))
            print(f"\nObject sort completed, time: {object_stats.execution_time:.4f} seconds")

    test_realtime_predict_logic(
        data_sizes=training_sizes_for_tests,
        strategy=SortStrategy.AUTO.value,
        profile=False,
        cache_size=2000,
        adaptive_threshold=0.8,
        use_ml_prediction=True,
        external_sort_threshold=1000000,
        eco_mode=False,
        priority_mode="speed",
        deduplicate_sort=False,
        data_type="number",
        log_level=logging.INFO
    )

    sorter = EnhancedHyperionSort(
        strategy=SortStrategy.AUTO,
        profile=False,
        cache_size=2000,
        adaptive_threshold=0.8,
        use_ml_prediction=False,
        external_sort_threshold=1000000,
        eco_mode=True,
        priority_mode="memory",
        deduplicate_sort=False,
        data_type="number",
        log_level=logging.INFO
    )

    priority_data = np.random.randint(0, 1000, size=1_000_000)
    sorted_priority_arr, priority_stats = asyncio.run(sorter.sort(priority_data))
    print(f"\nPriority sort mode completed, total time: {priority_stats.execution_time:.4f}")
    print(f"Priority sort mode items: {len(sorted_priority_arr):,} items")

    sorter = EnhancedHyperionSort(
        strategy=SortStrategy.STREAM,
        profile=False,
        cache_size=2000,
        adaptive_threshold=0.8,
        stream_mode=True,
        use_ml_prediction=False,
        data_type="number",
        log_level=logging.INFO
    )

    def streaming_data():
        for i in range(1000):
            yield np.random.randint(0, 100, size=1000)

    sorted_stream, streaming_stats = asyncio.run(sorter.sort(streaming_data()))
    print(f"\nStreaming sort completed, total time: {streaming_stats.execution_time:.4f}")
    print(f"Streaming sort processed items : {streaming_stats.items_processed:,}")

    sorter = EnhancedHyperionSort(
        strategy=SortStrategy.STREAMING_HYBRID_SORT,
        profile=False,
        cache_size=2000,
        adaptive_threshold=0.8,
        stream_mode=True,
        use_ml_prediction=False,
        data_type="number",
        log_level=logging.INFO
    )

    sorted_stream, streaming_stats = asyncio.run(sorter.sort(streaming_data()))
    print(f"\nStreaming hybrid sort completed, total time: {streaming_stats.execution_time:.4f}")
    print(f"Streaming hybrid processed items : {streaming_stats.items_processed:,}")

    if profile_execution:
        sorter = EnhancedHyperionSort(
            strategy=SortStrategy.AUTO,
            profile=False,
            cache_size=2000,
            adaptive_threshold=0.8,
            use_ml_prediction=False,
            external_sort_threshold=1000000,
            eco_mode=False,
            priority_mode="speed",
            deduplicate_sort=False,
            data_type="number",
            log_level=logging.INFO
        )
        data_arr = np.random.randint(0, 1000, size=1_000_000)
        start_time_profile = time.perf_counter()
        sorter._benchmark_on_the_fly(data_arr, Algorithm.QUICKSORT)
        sorter._benchmark_on_the_fly(data_arr, Algorithm.TIMSORT)
        sorter._benchmark_on_the_fly(data_arr, Algorithm.INTROSORT)
        stats = sorter.metrics.get_summary()

        print(f"\nBenchmark on the fly completed using profiler - this time check method overhead: total {time.perf_counter() - start_time_profile}")
        print(f"Benchmark stats: {stats}")

    pytest.main()