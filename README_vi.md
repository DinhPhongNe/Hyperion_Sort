# Enhanced Hyperion Sort: Thư viện sắp xếp thông minh và năng động

Repository này chứa một phiên bản nâng cấp và linh hoạt của các thuật toán sắp xếp, được thiết kế để đạt hiệu suất tối ưu trên nhiều loại và kích thước dữ liệu khác nhau. Nó bao gồm các tính năng như lựa chọn chiến lược thích ứng, xử lý luồng dữ liệu, đa luồng, dự đoán được tăng cường bởi machine learning, và nhiều tính năng khác, tạo nên một giải pháp sắp xếp linh hoạt và mạnh mẽ.

[ [Tiếng Anh](https://github.com/DinhPhongNe/Hyperion_Sort/blob/main/README.md)] - [ [Tiếng Việt](https://github.com/DinhPhongNe/Hyperion_Sort/blob/main/README_vi.md) ]

## Tính năng

*   **Chiến lược sắp xếp thích ứng:**
    *   Tự động chọn thuật toán sắp xếp phù hợp nhất (ví dụ: Quicksort, Mergesort, Heapsort, Timsort, Introsort, Radix Sort, External Merge Sort, Counting Sort, Quickselect,...) dựa trên đặc điểm dữ liệu (kích thước, phân phối,...) để đạt hiệu suất tối ưu.
    *   Các chiến lược dự phòng đảm bảo tính ổn định và độ tin cậy ngay cả khi chiến lược chính gặp vấn đề.
*   **Đa luồng và xử lý song song:**
    *   Tận dụng `ThreadPoolExecutor` và `ProcessPoolExecutor` để tối đa hóa việc sử dụng CPU, tận dụng các nhân có sẵn cho các tác vụ sắp xếp song song.
    *   Cơ chế cân bằng tải động để sử dụng tài nguyên tốt hơn.
*   **Hỗ trợ dữ liệu luồng:**
    *   Sắp xếp hiệu quả các luồng dữ liệu mà không cần tải toàn bộ vào bộ nhớ bằng `StreamProcessor` tùy chỉnh, với hồi quy tuyến tính để thay đổi kích thước chunk động.
    *   Hỗ trợ sắp xếp tăng dần lưu kết quả trung gian dưới dạng tệp, với cơ chế điều chỉnh kích thước chunk thích ứng. .
*   **Quản lý bộ nhớ:**
    *   Sử dụng chiến lược quản lý block cho các tập dữ liệu lớn, chia chúng thành các block nhỏ hơn, được gộp lại sau khi xử lý.
    *   Cơ chế cache thích ứng để tối ưu hóa hiệu suất.
    *   Kỹ thuật tiết kiệm bộ nhớ để giảm thiểu việc sử dụng bộ nhớ khi sắp xếp.
*   **Dự đoán Machine Learning:**
    *   Tích hợp TensorFlow, Catboost, Lightgbm, Xgboost để dự đoán chiến lược sắp xếp tối ưu dựa trên đặc điểm dữ liệu sử dụng mô hình ML đã được huấn luyện.
    *   Cơ chế phản hồi để cải thiện dự đoán của mô hình theo thời gian.
*   **Kiểm tra dữ liệu:**
    *   Thực hiện kiểm tra dữ liệu trước khi bắt đầu sắp xếp để đảm bảo tính toàn vẹn bằng cách kiểm tra kiểu dữ liệu và các giá trị vô cùng hoặc NaN.
    *   Hỗ trợ các kiểu dữ liệu khác nhau: số (số nguyên, số thực), chuỗi (với so sánh độ dài), đối tượng (bất kỳ kiểu dữ liệu nào, dựa trên khóa đầu tiên tìm thấy).
*   **Kỹ thuật nâng cao:**
    *   Cơ chế nén cho cả giảm dữ liệu và tốc độ, bằng cách nén với thuật toán LZ4.
    *   Điều chỉnh kích thước chunk thích ứng để sử dụng bộ nhớ hiệu quả và ngăn chặn tắc nghẽn.
*   **Đo lường và đánh giá hiệu năng:**
    *   Cung cấp đầu ra chi tiết bao gồm thời gian thực thi, dữ liệu sử dụng CPU, tiêu thụ bộ nhớ và các thông số hữu ích khác.
    *   Bao gồm module benchmark để kiểm tra các tham số sắp xếp khác nhau với các phân phối dữ liệu khác nhau.
    *   Phân tích hiệu suất thời gian thực bao gồm CPU, bộ nhớ và các chỉ số dự đoán để nâng cao việc ra quyết định.
*   **Ghi log & Giám sát:**
    *   Ghi log chi tiết để theo dõi luồng thực thi, lựa chọn chiến lược và dữ liệu hiệu suất thông qua module logging của Python.
    *   Bảng điều khiển hiệu suất thời gian thực được tích hợp, hiển thị CPU, bộ nhớ và các chỉ số khác.

## Cách sử dụng

### Cài đặt

1.  Clone repository này:

    ```bash
    git clone https://github.com/DinhPhongNe/Hyperion_Sort
    ```
2.  Cài đặt các dependencies từ `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Sắp xếp cơ bản

```python
from HyperionSort import EnhancedHyperionSort, SortStrategy
import numpy as np


sorter = EnhancedHyperionSort(strategy=SortStrategy.AUTO)
arr = np.random.randint(0, 100, size=1000)
sorted_arr, stats = asyncio.run(sorter.sort(arr))

print("Sorted array:", sorted_arr)
print("Sort statistics:", stats)
```

### Sắp xếp luồng dữ liệu
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

### Sắp xếp đối tượng

```python
from HyperionSort import EnhancedHyperionSort, SortStrategy
import asyncio

data = [{"name": "bob", "age": 30}, {"name": "alice", "age":25}]
sorter = EnhancedHyperionSort(data_type="object")
sorted_arr, stats  = asyncio.run(sorter.sort(data))
print("Sorted array:", sorted_arr)
print("Sort statistics:", stats)
```

### Sử dụng nâng cao
* Chỉ định chiến lược: Khởi tạo EnhancedHyperionSort với một SortStrategy cụ thể cho phương pháp sắp xếp cụ thể.

* Cấu hình Workers: Điều khiển số lượng workers song song bằng tham số n_workers.

* Kiểu dữ liệu tùy chỉnh: đặt data_type là "number", "string" hoặc "object".

* Profiling: Sử dụng profile=True để phân tích hiệu suất chi tiết.

* Chế độ ưu tiên: đặt priority_mode để tập trung vào "speed" (tốc độ), "memory" (bộ nhớ) hoặc "reliability" (độ tin cậy).

* Chế độ tiết kiệm: Sử dụng eco_mode = True để bật ghi log động để sử dụng tài nguyên hiệu quả hơn.

* Các tham số cấu hình khác được giải thích chi tiết trong mã nguồn.

### Đánh giá hiệu năng
* Dự án bao gồm công cụ benchmark để kiểm tra tốc độ và hiệu suất sắp xếp.

1. Đặt tham số benchmark = True để bật module benchmark
```python
sorter = EnhancedHyperionSort(strategy=SortStrategy.AUTO, benchmark=True)
```

2. Chạy module benchmark với các số đã định nghĩa trước (hoặc bạn có thể thay đổi chúng theo ý muốn):
```python
test_sizes = [100_000, 1_000_000, 10_000_000, 20_000_000]
benchmark_results = benchmark(sorter=sorter, sizes=test_sizes, runs=3, save_results=True)
```

1. Các test mẫu bổ sung có thể được xem trong khối code if __name__ == "__main__":

## Cấu hình

`EnhancedHyperionSort` có thể được cấu hình bằng cách truyền các đối số trong quá trình khởi tạo, chẳng hạn như:

* `strategy`: Chọn từ `SortStrategy.AUTO`, `SortStrategy.PARALLEL`, `SortStrategy.MEMORY_EFFICIENT`, và các chiến lược khác để chỉ định hoặc để nó tự động chọn chiến lược sắp xếp.
* `n_workers`: Số lượng luồng/tiến trình được sử dụng cho xử lý song song. Tự động chọn số lượng nhân dựa trên máy của bạn.
* `chunk_size`: Kích thước của các chunk được sử dụng khi làm việc với luồng dữ liệu.
* `cache_size`: Kích thước cache được sử dụng trong thuật toán.
* `adaptive_threshold`: Tham số động cho việc điều chỉnh kích thước cache và các quy trình khác.
* `stream_mode`: Bật chế độ sắp xếp luồng.
* `block_size`: Kích thước block cho sắp xếp mảng lớn.
* `profile`: Bật profiler (cho mục đích gỡ lỗi).
* `use_ml_prediction`: Bật mô hình ML để dự đoán chiến lược.
* `compression_threshold`: Ngưỡng kích thước dữ liệu để sử dụng nén.
* `external_sort_threshold`: Ngưỡng kích thước dữ liệu để sử dụng sắp xếp ngoại vi.
* `duplicate_ratio`: Phát hiện số lượng bản sao dữ liệu để xác định xem `counting_sort` có phù hợp không.
* `eco_mode`: Bật hoặc tắt ghi log động.
* `priority_mode`: Chọn "speed" (tốc độ), "memory" (bộ nhớ) hoặc "reliability" (độ tin cậy).
* `deduplicate_sort`: Sắp xếp mảng sau khi loại bỏ các mục trùng lặp.
* `service_mode`: Chế độ được tạo cho máy chủ để không lưu giữ lịch sử các lần sắp xếp trước đó.
* `data_type`: Kiểu dữ liệu là `number`, `string` hoặc `object`.

## Todo (Cải tiến trong tương lai)

* Nâng cao mô hình machine learning để phân loại dữ liệu tốt hơn.
* Quản lý bộ nhớ mạnh mẽ hơn để giảm overhead trên các lần sắp xếp lớn.
* Thêm nhiều biểu đồ hóa cho các số liệu.
* Triển khai thêm các thuật toán sắp xếp.
* Tái cấu trúc codebase để cải thiện tính ổn định.
* Cải thiện tài liệu bằng chú thích code.
