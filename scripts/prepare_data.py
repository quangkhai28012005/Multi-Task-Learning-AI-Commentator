"""Script to prepare training data from sample data."""

import json
import random
from pathlib import Path
from src.data_processing.data_loader import DataLoader
from src.data_processing.preprocessing import create_samples, split_data


def generate_synthetic_data(num_samples=100):
    """Generate synthetic football match data.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of event dictionaries
    """
    action_types = ['goal', 'pass', 'tackle', 'foul', 'corner', 'free_kick',
                   'throw_in', 'substitution', 'yellow_card', 'red_card']
    
    teams = ['Team A', 'Team B']
    players = [
        'Player A1', 'Player A2', 'Player A3', 'Player A4', 'Player A5',
        'Player B1', 'Player B2', 'Player B3', 'Player B4', 'Player B5'
    ]
    
    sentiments = ['positive', 'neutral', 'negative']
    
    descriptions = {
        'goal': [
            'Cầu thủ sút chính xác vào góc tường',
            'Bàn thắng từ cú sút xa tuyệt đẹp',
            'Ghi bàn sau pha phát nhanh',
            'Bàn thắng từ đầu bóng',
            'Bàn thắng từ quả đá phạt'
        ],
        'pass': [
            'Đường chuyền chính xác cho đồng đội',
            'Đường chuyền ngang sân bền vững',
            'Chia bóng sau pha đóng chặt',
            'Đường chuyền dọc sân hiệu quả'
        ],
        'tackle': [
            'Cướp bóng sạch sẽ ở giữa sân',
            'Pha phòng chống rất khéo léo',
            'Hoàn tất pha cắt bóng tuyệt vời',
            'Cướp bóng mạnh mẽ từ đối thủ'
        ],
        'foul': [
            'Phạm lỗi kỹ thuật với đối thủ',
            'Kiếm được thẻ vàng từ trọng tài',
            'Phạm lỗi sau pha bóng',
            'Vi phạm trong vòng cấm'
        ],
        'corner': [
            'Sút phạt góc cho đội nhà',
            'Thực hiện quả góc từ bên phải',
            'Quả góc đưa bóng vào vòng cấm'
        ]
    }
    
    commentaries = {
        'goal': 'Bàn thắng! Cầu thủ vừa mở tỷ số cho đội nhà!',
        'pass': 'Đường chuyền chính xác!',
        'tackle': 'Pha cướp bóng rất sạch!',
        'foul': 'Lỗi được ghi nhận!',
        'corner': 'Quả góc được thực hiện!'
    }
    
    data = []
    for i in range(num_samples):
        match_id = f"match_{(i // 10) + 1:03d}"
        timestamp = f"{random.randint(0, 90)}:{random.randint(0, 59):02d}"
        action_type = random.choice(action_types)
        team = random.choice(teams)
        player = random.choice(players)
        
        description = random.choice(descriptions.get(action_type, ['Sự kiện trong trận đấu']))
        commentary = commentaries.get(action_type, 'Bình luận')
        sentiment = 'positive' if action_type == 'goal' else 'negative' if action_type == 'foul' else random.choice(sentiments)
        
        importance_score = random.uniform(5, 10) if action_type in ['goal', 'yellow_card'] else random.uniform(3, 8)
        
        event = {
            'match_id': match_id,
            'timestamp': timestamp,
            'action_type': action_type,
            'team': team,
            'player': player,
            'description': description,
            'commentary': commentary,
            'sentiment': sentiment,
            'importance_score': round(importance_score, 1)
        }
        data.append(event)
    
    return data


def prepare_data():
    """Prepare training, validation, and test data."""
    print("Preparing training data...\n")
    
    # Load sample data
    sample_data_path = 'data/samples/sample_data.json'
    sample_data = DataLoader.load_json(sample_data_path)
    print(f"Loaded {len(sample_data)} sample events")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(150)
    print(f"Generated {len(synthetic_data)} synthetic events")
    
    # Combine data
    all_data = sample_data + synthetic_data
    print(f"Total events: {len(all_data)}")
    
    # Create DataFrame
    df = create_samples(all_data)
    
    # Split data
    train_df, val_df, test_df = split_data(df, train_ratio=0.7, val_ratio=0.15)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Create output directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Save data
    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    test_data = test_df.to_dict('records')
    
    DataLoader.save_json(train_data, 'data/processed/train.json')
    DataLoader.save_json(val_data, 'data/processed/val.json')
    DataLoader.save_json(test_data, 'data/processed/test.json')
    
    print(f"\n✓ Data saved:")
    print(f"  - data/processed/train.json")
    print(f"  - data/processed/val.json")
    print(f"  - data/processed/test.json")


if __name__ == '__main__':
    prepare_data()
