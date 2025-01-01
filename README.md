# Enhanced Hyperion Sort: A Smart and Dynamic Sorting Library

This repository contains an enhanced and dynamic implementation of various sorting algorithms, designed for optimal performance across a wide range of data types and sizes. It includes adaptive strategy selection, stream processing, multi-threading, machine learning enhanced predictions, and more, to provide a flexible and powerful sorting solution.

[ [English](https://github.com/DinhPhongNe/Hyperion_Sort/blob/main/README.md)] - [ [Vietnamses](https://github.com/DinhPhongNe/Hyperion_Sort/blob/main/README_vi.md) ]
## Features

*   **Adaptive Sorting Strategy:**
    *   Automatically selects the most appropriate sorting algorithm (e.g., Quicksort, Mergesort, Heapsort, Timsort, Introsort, Radix Sort, External Merge Sort, Counting Sort, Quickselect, etc.) based on data characteristics (size, distribution, etc.) for optimal performance.
    *   Fallback strategies ensure stability and reliability even when the primary strategy encounters issues.
*   **Multi-threading and Parallel Processing:**
    *   Leverages `ThreadPoolExecutor` and `ProcessPoolExecutor` to maximize CPU utilization, utilizing available cores for parallel sorting tasks.
    *   Dynamic load balancing mechanism for better resource usage.
*   **Streaming Data Support:**
    *   Efficiently sorts data streams without loading them entirely in memory using our custom `StreamProcessor`, with linear regression to dynamically change chunk sizes.
    *   Supports incremental sorting that saves intermediate results as file, with adaptive chunk size mechanism.
*   **Memory Management:**
    *   Employs block management strategies for large datasets, splitting them into smaller blocks, merged after processing.
    *   Adaptive caching mechanism for performance optimizations.
    *   Memory-efficient techniques to conserve memory usage during sort.
*   **Machine Learning Prediction:**
    *   Integration of TensorFlow to predict the optimal sorting strategy based on the data characteristics using a trained ML model.
    *   A feedback loop mechanism to improve model prediction over time.
*   **Data Validation:**
    *   Performs data validation before start sort to ensure data integrity by checking data type and if it has infinity or NaN values.
    *   Supports different data types: number (integers, floats), string (with length comparison), object (any data type, based on the first key found).
*   **Advanced Techniques:**
    *   Compression mechanisms for both data reduction and speed, by compressing with LZ4 algorithms.
    *   Adaptive chunk sizing to use memory effectively and prevent bottlenecks.
*   **Metrics and Benchmarking:**
    *   Provides detailed output including execution times, CPU usage data, memory consumption, and other useful metrics.
    *   Includes benchmark module to test various sorting parameters with different data distributions.
    *   Real-time performance analysis including CPU, memory, and predictive metrics to enhance decision-making.
*   **Logging & Monitoring:**
    *   Detailed logging to track execution flow, strategy selections, and performance data through Python's logging module.
    *   Real-time performance dashboard included, shows CPU, memory, and other metrics.

## Usage

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/DinhPhongNe/Hyperion_Sort
    ```
2.  Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Note: if you do not have requirements.txt create it by running `pip freeze >> requirements.txt`. Ensure you have python, numpy, tensorflow, and other depedencies installed already.

### Basic Sorting

```python
from HyperionSort import EnhancedHyperionSort, SortStrategy
import numpy as np


sorter = EnhancedHyperionSort(strategy=SortStrategy.AUTO)
arr = np.random.randint(0, 100, size=1000)
sorted_arr, stats = asyncio.run(sorter.sort(arr))

print("Sorted array:", sorted_arr)
print("Sort statistics:", stats)
```

### Stream Sorting
```python
from HyperionSort import EnhancedHyperionSort, SortStrategy
import numpy as np
import asyncio


def my_data_stream():
    for _ in range(1000):
        yield np.random.randint(0, 100, size = 10)


sorter = EnhancedHyperionSort(strategy=SortStrategy.STREAM, stream_mode=True)
sorted_arr, stats  = asyncio.run(sorter.sort(my_data_stream()))
print("Sorted array:", sorted_arr)
print("Sort statistics:", stats)
```

### Object sorting

```python
from HyperionSort import EnhancedHyperionSort, SortStrategy
import asyncio

data = [{"name": "bob", "age": 30}, {"name": "alice", "age":25}]
sorter = EnhancedHyperionSort(data_type="object")
sorted_arr, stats  = asyncio.run(sorter.sort(data))
print("Sorted array:", sorted_arr)
print("Sort statistics:", stats)
```

### Advanced Usage
* Specify Strategy: Initialize EnhancedHyperionSort with a specific SortStrategy for a particular sorting method.

* Configure Workers: Control the number of parallel workers using the n_workers parameter.

* Custom Data Types: set data_type to be "number", "string" or "object".

* Profiling: Use profile=True for detailed performance analysis.

* Priority Mode: set priority_mode to focus on either "speed", "memory" usage or "reliability".

* Eco Mode: Use the eco_mode = True flag to enable dynamic logging to use resources more efficiently.

* Other configuration parameters are explained in great detail through the source code.

### Benchmarking
* The project includes a benchmarking tool to test sorting speeds and performance.

1. Set the parameter benchmark = True to enable the benchmarking module
```python
sorter = EnhancedHyperionSort(strategy=SortStrategy.AUTO, benchmark=True)
```

2. Run the benchmark module with your pre-defined numbers (or you can change them as you like):
```python
test_sizes = [100_000, 1_000_000, 10_000_000, 20_000_000]
benchmark_results = benchmark(sorter=sorter, sizes=test_sizes, runs=3, save_results=True)
```

1. Additional sample test can be seen in if __name__ == "__main__": block of code.

## Configuration

The `EnhancedHyperionSort` can be configured by passing the arguments during its initialization, such as:

*   `strategy`: Choose from `SortStrategy.AUTO`, `SortStrategy.PARALLEL`, `SortStrategy.MEMORY_EFFICIENT`, and others to specify or let it automatically choose a sorting strategy.
*   `n_workers`: The number of threads/processes used for parallel processing. It automatically chooses number of cores based on your machine.
*   `chunk_size`: The size of the chunks used when working with data streams.
*   `cache_size`: The size of cache used in algorithm.
*   `adaptive_threshold`: Dynamic parameter for cache sizing and other processes.
*   `stream_mode`: Enables stream sorting mode.
*   `block_size`: The block size for large array sort.
*   `profile`: Enable profiler (for debug purposes).
*   `use_ml_prediction`: Enable ML model to predict the strategy.
*   `compression_threshold`: Threshold data size to use compression.
*   `external_sort_threshold`: Threshold data size to use external sort.
*   `duplicate_ratio`: Detects how many duplicates a data has to determine if `counting_sort` is adequate.
*   `eco_mode`: Whether to enable or disable dynamic logging.
*   `priority_mode`: Choose either "speed", "memory" or "reliability".
*   `deduplicate_sort`: Sort the array after deduplicating its entries.
*  `service_mode`: A mode created for a server to not keep a history of past sorts.
*   `data_type`: Data type either `number`, `string` or `object`.

## To-Do (Future Improvements)

*   Enhancement to the machine learning model to better classify data.
*   More robust memory management to reduce overhead on large sorts.
*   More data visualization of metrics.
*   Implement more sorting algorithms.
*   Refactor code base to improve stability.
*   Improve documentation by code annotation.
