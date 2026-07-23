# Multi-Task Learning AI Commentator

## Hệ thống AI phân tích bình luận trận đấu bóng đá

Xây dựng hệ thống Multi-Task Learning AI Commentator hỗ trợ bình luận, phân tích, đánh giá các hành vi, hoạt động diễn ra trong một trận đấu bóng đá.

## 🎯 Tính Năng Chính

### 1. Nhận Dạng Hành Động (Action Recognition)
- Phân loại các loại hành động trong trận đấu
- Hỗ trợ: ghi bàn, đường chuyền, cướp bóng, phạm lỗi, v.v.

### 2. Sinh Bình Luận (Commentary Generation)
- Tạo bình luận phân tích tự động
- Bình luận phù hợp với bối cảnh trận đấu

### 3. Phân Tích Cảm Xúc (Sentiment Analysis)
- Nhận dạng cảm xúc: tích cực, trung lập, tiêu cực
- Dựa trên loại sự kiện và bối cảnh

### 4. Đánh Giá Sự Kiện (Event Importance)
- Đánh giá mức độ quan trọng (0-10)
- Dựa trên loại hành động và tác động

## 🚀 Bắt Đầu Nhanh

### Yêu Cầu
- Python 3.8+
- pip

### Cài Đặt

```bash
# Clone repository
git clone https://github.com/quangkhai28012005/Multi-Task-Learning-AI-Commentator.git
cd Multi-Task-Learning-AI-Commentator

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Setup environment
python setup_env.py
```

### Test Nhanh

```bash
python main.py --test
```

### Demo Phân Tích

```bash
# Phân tích dữ liệu mẫu
python main.py --demo

# Phân tích một sự kiện
python main.py --demo --demo-mode single
```

## 📝 Sử Dụng

### Phân Tích Một Sự Kiện

```python
from src.inference.commentator import AICommentator

# Khởi tạo
commentator = AICommentator()

# Định nghĩa sự kiện
event = {
    'action_type': 'goal',
    'team': 'Manchester United',
    'player': 'Bruno Fernandes',
    'description': 'Cầu thủ sút chính xác vào góc tường'
}

# Phân tích
result = commentator.analyze(event)

# Kết quả
print(f"Action: {result['action_type']}")
print(f"Commentary: {result['commentary']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Importance: {result['importance_score']}/10")
```

### Phân Tích Hàng Loạt

```python
from src.inference.commentator import AICommentator
from src.data_processing.data_loader import DataLoader

# Tải dữ liệu
events = DataLoader.load_json('data/samples/sample_data.json')

# Khởi tạo
commentator = AICommentator()

# Phân tích
results = commentator.batch_analyze(events)

# Xử lý kết quả
for event, result in zip(events, results):
    print(f"{event['timestamp']}: {result['commentary']}")
```

## 📚 Các Lệnh Chính

| Lệnh | Mô Tả |
|------|-------|
| `python main.py --test` | Kiểm thử nhanh |
| `python main.py --demo` | Demo phân tích |
| `python main.py --prepare-data` | Chuẩn bị dữ liệu huấn luyện |
| `python main.py --train` | Huấn luyện mô hình |
| `python main.py --evaluate` | Đánh giá mô hình |
| `pytest tests/` | Chạy unit tests |

## 📁 Cấu Trúc Dự Án

```
.
├── src/                    # Mã nguồn chính
│   ├── models/            # Định nghĩa mô hình
│   ├── data_processing/   # Xử lý dữ liệu
│   ├── inference/         # Engine phân tích
│   ├── train.py           # Script huấn luyện
│   ├── evaluate.py        # Script đánh giá
│   └── demo.py            # Script demo
│
├── data/                  # Dữ liệu
│   ├── samples/           # Dữ liệu mẫu
│   ├── raw/               # Dữ liệu thô
│   └── processed/         # Dữ liệu xử lý
│
├── models/                # Mô hình đã huấn luyện
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── scripts/               # Các script tiện ích
├── config/                # Tệp cấu hình
├── main.py                # Entry point chính
├── requirements.txt       # Dependencies
└── README.md              # File này
```

## 🔧 Cấu Hình

Sửa `config/training_config.yaml` để điều chỉnh:
- Batch size
- Learning rate
- Số epoch
- Số workers
- Device (cuda/cpu)

## 📊 Định Dạng Dữ Liệu

Sự kiện phải có định dạng JSON:

```json
{
  "match_id": "match_001",
  "timestamp": "45:30",
  "action_type": "goal",
  "team": "Team A",
  "player": "Player Name",
  "description": "Mô tả sự kiện",
  "sentiment": "positive",
  "importance_score": 9.5
}
```

## 🤖 Kiến Trúc Mô Hình

```
Input (Action + Description)
         ↓
    BERT Encoder (Shared)
         ↓
    ┌─────┬─────┬─────┐
    ↓     ↓     ↓     ↓
  Task1 Task2 Task3 Task4
  (Action) (Commentary) (Sentiment) (Importance)
```

## ✅ Các Tác Vụ

1. **Task 1**: Nhận dạng hành động (10 loại)
2. **Task 2**: Sinh bình luận
3. **Task 3**: Phân tích cảm xúc (3 lớp)
4. **Task 4**: Đánh giá sự kiện (hồi quy)

## 📈 Hiệu Năng

- Inference: ~0.5-2 giây/sự kiện (tùy hardware)
- Batch processing: Tối ưu cho xử lý hàng loạt
- GPU acceleration: Hỗ trợ CUDA

## 🧪 Kiểm Thử

```bash
# Chạy quick test
python scripts/quick_test.py

# Chạy unit tests
pytest tests/

# Chạy tất cả tests
bash run_tests.sh
```

## 📖 Tài Liệu Thêm

- [Installation Guide](INSTALL.md) - Hướng dẫn chi tiết cài đặt
- [User Guide](USAGE.md) - Hướng dẫn sử dụng
- [Demo Notebook](notebooks/Demo.ipynb) - Jupyter notebook demo

## 🤝 Đóng Góp

Chúng tôi chào đón các đóng góp! Vui lòng:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - xem [LICENSE](LICENSE) file

## 👤 Tác Giả

**Quang Khai** - [GitHub](https://github.com/quangkhai28012005)

## 📧 Liên Hệ

Nếu có câu hỏi hoặc đề xuất, vui lòng:
- Mở issue trên GitHub
- Gửi email
- Liên hệ qua GitHub discussions

## 🙏 Cảm Ơn

- PyTorch team
- Hugging Face (Transformers library)
- Tất cả những người đóng góp

---

**Made with ❤️ for Football Analysis**
