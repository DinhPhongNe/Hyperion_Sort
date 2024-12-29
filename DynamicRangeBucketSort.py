from typing import List, Union, Optional, Tuple, Dict, Any, Callable
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math
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
from typing import Generator
import multiprocessing as mp
import random

# Thêm logging chi tiết hơn [[2](https://cybersoft.edu.vn/ngay-4-nghe-thuat-toi-uu-hoa-khi-code-dat-dinh-cao-hoan-my/)]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(processName)s'
)
logger = logging.getLogger(__name__)

class SortStrategy(Enum):
    AUTO = "auto"
    PARALLEL = "parallel" 
    MEMORY_EFFICIENT = "memory_efficient"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    STREAM = "stream" 

class Algorithm(Enum):
    QUICKSORT = "quicksort"
    MERGESORT = "mergesort"
    HEAPSORT = "heapsort" 
    TIMSORT = "timsort"
    INTROSORT = "introsort"
    

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
        
    def process_stream(self, data_stream: Generator) -> Generator:
        for item in data_stream:
            with self._lock:
                self.buffer.append(item)
                if len(self.buffer) >= self.chunk_size:
                    yield sorted(self.buffer)
                    self.buffer.clear()
        
        if self.buffer:
            yield sorted(self.buffer)

class MetricsCollector:
    def __init__(self):
        self.metrics = []
        self.start_time = time.perf_counter()

    def record(self, metric_name: str, value: Any):
        self.metrics.append({
            'name': metric_name,
            'value': value,
            'timestamp': time.perf_counter() - self.start_time
        })

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_duration': time.perf_counter() - self.start_time,
            'metrics': self.metrics
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

class EnhancedDynamicRangeBucketSort:
    def __init__(
        self,
        strategy: SortStrategy = SortStrategy.AUTO,
        n_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        profile: bool = False,
        cache_size: int = 1000,
        adaptive_threshold: float = 0.7,
        stream_mode: bool = False
    ):
        self.strategy = strategy
        self.profile = profile
        self.n_workers = n_workers or max(1, psutil.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.cache = CacheManager(cache_size)
        self.metrics = MetricsCollector()
        self.adaptive_threshold = adaptive_threshold
        self.stream_mode = stream_mode
        self.stream_processor = StreamProcessor() if stream_mode else None
        self._setup_logging()
        
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
            
            return np.concatenate([
                self._introsort(left, max_depth - 1),
                middle,
                self._introsort(right, max_depth - 1)
            ])
    
    def _ninther(self, arr: npt.NDArray) -> float:
        """Improved pivot selection using ninther method"""
        if len(arr) < 9:
            return np.median(arr)
            
        thirds = len(arr) // 3
        medians = [
            np.median([arr[i], arr[i + thirds], arr[i + 2*thirds]])
            for i in range(3)
        ]
        return np.median(medians)
        
    def _insertion_sort(self, arr: npt.NDArray) -> npt.NDArray:
        """Optimized insertion sort for small arrays"""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
        
    def _heapsort(self, arr: npt.NDArray) -> npt.NDArray:
        """Implementation of heapsort"""
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

    def _stream_sort(self, data_stream: Generator) -> Generator:
        """New method for handling streaming data"""
        return self.stream_processor.process_stream(data_stream)

    def sort(
        self, 
        arr: Union[List[float], npt.NDArray, Generator],
        stream: bool = False
    ) -> Union[Tuple[npt.NDArray, SortStats], Generator]:
        self.start_time = time.perf_counter()
        self.logger.info("Starting sort operation...")
        
        if stream or isinstance(arr, Generator):
            return self._stream_sort(arr)
            
        if self.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            if isinstance(arr, list):
                arr = np.array(arr, dtype=np.float64)
                
            strategy = (
                self._choose_optimal_strategy(arr)
                if self.strategy == SortStrategy.AUTO
                else self.strategy
            )
            
            self.logger.info(f"Selected strategy: {strategy.value}")
            
            with performance_tracker():
                if strategy == SortStrategy.MEMORY_EFFICIENT:
                    result = self._memory_efficient_sort(arr)
                elif strategy == SortStrategy.HYBRID:
                    result = (self._hybrid_sort(arr),
                             self._calculate_stats(arr, "hybrid", Algorithm.INTROSORT.value))
                elif strategy == SortStrategy.ADAPTIVE:
                    result = self._adaptive_sort(arr)
                elif strategy == SortStrategy.STREAM:
                    return self._stream_sort(arr)
                else:
                    result = self._parallel_sort(arr)
                    
            if self.profile:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats()
                self.logger.debug(f"Profile results:\n{s.getvalue()}")
                
            self.logger.info("Sort operation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during sorting: {e}", exc_info=True)
            return arr, self._calculate_stats(arr, "failed", "none")
        
    def _setup_logging(self):
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
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

    def _adaptive_chunk_size(self, arr_size: int, itemsize: int) -> int:
        available_memory = psutil.virtual_memory().available
        total_size = arr_size * itemsize
        cpu_count = psutil.cpu_count()
        
        l3_cache = psutil.cpu_count() * 2**20
        
        if total_size < l3_cache:
            return min(arr_size, 10000)
            
        optimal_chunks = max(
            cpu_count,
            int(total_size / (available_memory * self.adaptive_threshold))
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
            mask = (arr >= pivots[i-1]) & (arr < pivots[i])
            partition = arr[mask]
            if len(partition) > 0:
                partitions.append(partition)
                
        return partitions

    def _hybrid_sort(self, arr: npt.NDArray) -> npt.NDArray:
        if len(arr) < 1000:
            return np.sort(arr)
            
        partitions = self._advanced_partition(arr)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            sorted_partitions = list(executor.map(np.sort, partitions))
            
        return np.concatenate(sorted_partitions)

    def _merge_sorted_arrays(self, arrays: List[npt.NDArray]) -> npt.NDArray:
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
        n = len(arr)
        
        sample = arr[np.random.choice(n, min(1000, n), replace=False)]
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

    def _optimize_bucket_count(self, arr: npt.NDArray) -> int:
        n = len(arr)
        if n < 1000:
            return max(1, n // 10)
            
        cache_key = f"bucket_count_{n}_{arr.std():.2f}"
        if cached_value := self.cache.get(cache_key):
            return cached_value
            
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]
        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        
        if std_dev < range_size / 100:
            bucket_count = int(math.sqrt(n))
        else:
            density = n / range_size if range_size > 0 else 1
            bucket_count = int(min(
                math.sqrt(n) * (std_dev / range_size) * 2,
                n / math.log2(n)
            ))
            
        self.cache.put(cache_key, bucket_count)
        return bucket_count

    def _calculate_stats(self, arr: npt.NDArray, strategy: str, algorithm: str) -> SortStats:
        process = psutil.Process()
        
        performance = PerformanceMetrics(
            cpu_time=time.process_time(),
            wall_time=time.perf_counter() - self.start_time,
            memory_peak=process.memory_info().rss / (1024 * 1024),
            cache_hits=self.cache.hits,
            cache_misses=self.cache.misses,
            thread_count=len(process.threads()),
            context_switches=process.num_ctx_switches().voluntary
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
            optimization_history=self.metrics.metrics
        )

    def sort(self, arr: Union[List[float], npt.NDArray]) -> Tuple[npt.NDArray, SortStats]:
        self.start_time = time.perf_counter()
        self.logger.info("Starting sort operation...")
        
        if self.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            if isinstance(arr, list):
                arr = np.array(arr, dtype=np.float64)
                
            strategy = (
                self._choose_optimal_strategy(arr)
                if self.strategy == SortStrategy.AUTO
                else self.strategy
            )
            
            self.logger.info(f"Selected strategy: {strategy.value}")
            
            with performance_tracker():
                if strategy == SortStrategy.MEMORY_EFFICIENT:
                    result = self._memory_efficient_sort(arr)
                elif strategy == SortStrategy.HYBRID:
                    result = (self._hybrid_sort(arr), 
                             self._calculate_stats(arr, "hybrid", Algorithm.QUICKSORT.value))
                elif strategy == SortStrategy.ADAPTIVE:
                    result = self._adaptive_sort(arr)
                else:
                    result = self._parallel_sort(arr)
                    
            if self.profile:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats()
                self.logger.debug(f"Profile results:\n{s.getvalue()}")
                
            self.logger.info("Sort operation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during sorting: {e}", exc_info=True)
            return arr, self._calculate_stats(arr, "failed", "none")

    def _choose_optimal_strategy(self, arr: npt.NDArray) -> SortStrategy:
        n = len(arr)
        
        sample_size = min(1000, n)
        sample = arr[np.random.choice(n, sample_size, replace=False)]
        
        is_nearly_sorted = np.sum(np.diff(sample) < 0) < len(sample) * 0.1
        std_dev = np.std(sample)
        range_size = np.ptp(sample)
        memory_available = psutil.virtual_memory().available
        
        estimated_memory = n * arr.itemsize * 3
        
        if estimated_memory > memory_available * 0.7:
            return SortStrategy.MEMORY_EFFICIENT
        
        if is_nearly_sorted:
            return SortStrategy.ADAPTIVE
        
        if n > 1_000_000 and psutil.cpu_count() > 2:
            return SortStrategy.PARALLEL
        
        if std_dev < range_size / 100:
            return SortStrategy.HYBRID
            
        return SortStrategy.ADAPTIVE

    def _get_bucket_distribution(self, arr: npt.NDArray) -> List[int]:
        if len(arr) == 0:
            return []
            
        n_buckets = self._optimize_bucket_count(arr)
        bucket_ranges = np.linspace(arr.min(), arr.max(), n_buckets + 1)
        distribution = []
        
        for i in range(n_buckets):
            mask = (arr >= bucket_ranges[i]) & (arr < bucket_ranges[i + 1])
            distribution.append(int(np.sum(mask)))
            
        return distribution

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
            np.sort(chunk, out=chunk)
            result[i:i + chunk_size] = chunk
            
        return result, self._calculate_stats(arr, "memory_efficient", Algorithm.MERGESORT.value)

def benchmark(
    sorter: EnhancedDynamicRangeBucketSort,
    sizes: List[int],
    runs: int = 3,
    save_results: bool = True
) -> Dict[str, Any]:
    results = []
    
    for size in sizes:
        size_results = []
        for run in range(runs):
            logger.info(f"\nBenchmarking với {size:,} phần tử (Run {run + 1}/{runs}):")
            
            if run == 0:
                data = np.random.randint(0, size * 10, size=size)
            elif run == 1:
                data = np.sort(np.random.randint(0, size * 10, size=size))
                data[::100] = np.random.randint(0, size * 10, size=len(data[::100]))
            else:
                n_clusters = 10
                cluster_points = size // n_clusters
                clusters = []
                
                for i in range(n_clusters):
                    center = np.random.randint(0, size * 10)
                    cluster = np.random.normal(loc=center, 
                                            scale=size/100, 
                                            size=cluster_points)
                    clusters.append(cluster)
                
                data = np.concatenate(clusters).astype(np.int32)
            
            sorted_arr, stats = sorter.sort(data)
            
            is_sorted = np.all(sorted_arr[:-1] <= sorted_arr[1:])
            
            print(f"\n✨ Kết quả chạy {run + 1}:")
            print(f"✓ Đã sort xong!")
            print(f"⚡ Thời gian thực thi: {stats.execution_time:.4f} giây")
            print(f"⏱️ CPU time: {stats.performance.cpu_time:.4f} giây")
            print(f"💾 Bộ nhớ sử dụng: {stats.memory_usage:.2f} MB")
            print(f"🖥️ CPU usage: {stats.cpu_usage:.1f}%")
            print(f"🚀 Tốc độ xử lý: {stats.items_processed/stats.execution_time:,.0f} items/giây")
            print(f"🎯 Strategy: {stats.strategy_used}")
            print(f"🔄 Algorithm: {stats.algorithm_used}")
            print(f"📊 Cache hits/misses: {stats.performance.cache_hits}/{stats.performance.cache_misses}")
            print(f"✓ Kết quả đúng: {is_sorted}")
            
            result_data = {
                'size': size,
                'run': run,
                'distribution': ['random', 'nearly_sorted', 'clustered'][run],
                'time': stats.execution_time,
                'cpu_time': stats.performance.cpu_time,
                'memory': stats.memory_usage,
                'strategy': stats.strategy_used,
                'algorithm': stats.algorithm_used,
                'cache_hit_ratio': (stats.performance.cache_hits / 
                                  (stats.performance.cache_hits + stats.performance.cache_misses)
                                  if (stats.performance.cache_hits + stats.performance.cache_misses) > 0 
                                  else 0),
                'items_per_second': stats.items_processed/stats.execution_time,
                'is_sorted': is_sorted
            }
            
            size_results.append(result_data)
            
        avg_time = np.mean([r['time'] for r in size_results])
        std_time = np.std([r['time'] for r in size_results])
        
        print(f"\n📊 Thống kê cho {size:,} phần tử:")
        print(f"📈 Thời gian trung bình: {avg_time:.4f} ± {std_time:.4f} giây")
        print(f"🎯 Độ ổn định: {(1 - std_time/avg_time) * 100:.2f}%")
        
        results.extend(size_results)
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"\n💾 Đã lưu kết quả vào: {filename}")
    
    return {
        'results': results,
        'summary': {
            'total_runs': len(results),
            'sizes_tested': sizes,
            'best_performance': min(results, key=lambda x: x['time']),
            'worst_performance': max(results, key=lambda x: x['time'])
        }
    }
    
if __name__ == "__main__":
    sorter = EnhancedDynamicRangeBucketSort(
        strategy=SortStrategy.AUTO,
        profile=True,
        cache_size=2000,
        adaptive_threshold=0.8
    )
    
    test_sizes = [100_000, 1_000_000, 10_000_000]
    benchmark_results = benchmark(
        sorter=sorter,
        sizes=test_sizes,
        runs=3,
        save_results=True
    )
    def data_generator():
        for i in range(1000000):
            yield np.random.randint(0, 1000)
    
    sorter = EnhancedDynamicRangeBucketSort(
        strategy=SortStrategy.STREAM,
        profile=True,
        cache_size=2000,
        adaptive_threshold=0.8,
        stream_mode=True
    )
    
    for sorted_chunk in sorter.sort(data_generator(), stream=True):
        print(f"Sorted chunk size: {len(sorted_chunk)}")
        
    
    print("\n🌟 Kết quả tổng quan:")
    print(f"📊 Tổng số lần chạy: {benchmark_results['summary']['total_runs']}")
    print("\n🏆 Hiệu suất tốt nhất:")
    best = benchmark_results['summary']['best_performance']
    print(f"- Kích thước: {best['size']:,}")
    print(f"- Thời gian: {best['time']:.4f} giây")
    print(f"- Strategy: {best['strategy']}")
    print(f"- Distribution: {best['distribution']}")