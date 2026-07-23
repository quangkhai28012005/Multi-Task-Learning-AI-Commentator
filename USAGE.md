# Multi-Task Learning AI Commentator - User Guide

## Overview

This guide explains how to use the Multi-Task Learning AI Commentator system for analyzing football match events.

## Key Features

### 1. **Action Recognition**
Automatically identifies the type of action in a football match:
- Goal
- Pass
- Tackle
- Foul
- Corner
- Free kick
- Throw-in
- Substitution
- Yellow card
- Red card

### 2. **Commentary Generation**
Generates contextual commentary for each event based on the action type and teams involved.

### 3. **Sentiment Analysis**
Determines the emotional tone of events:
- **Positive**: Goals, successful plays
- **Neutral**: Regular passes, routine plays
- **Negative**: Fouls, yellow/red cards

### 4. **Event Evaluation**
Scores the importance of events from 0-10 based on their significance in the match.

## Usage Examples

### Example 1: Analyze a Single Event

```python
from src.inference.commentator import AICommentator

# Initialize commentator
commentator = AICommentator()

# Define an event
event = {
    'action_type': 'goal',
    'team': 'Manchester United',
    'player': 'Bruno Fernandes',
    'description': 'Cầu thủ thực hiện cú sút xa chính xác vào góc tường'
}

# Analyze event
result = commentator.analyze(event)

# Print results
print(f"Action: {result['action_type']}")
print(f"Commentary: {result['commentary']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Importance: {result['importance_score']}/10")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

### Example 2: Batch Analysis

```python
from src.inference.commentator import AICommentator
from src.data_processing.data_loader import DataLoader

# Load sample data
events = DataLoader.load_json('data/samples/sample_data.json')

# Initialize commentator
commentator = AICommentator()

# Analyze all events
results = commentator.batch_analyze(events)

# Process results
for event, result in zip(events, results):
    print(f"Match: {event['match_id']} at {event['timestamp']}")
    print(f"Analysis: {result['commentary']}")
    print()
```

### Example 3: Load Custom Data

```python
import json
from src.data_processing.data_loader import DataLoader

# Load JSON data
events = DataLoader.load_json('path/to/your/data.json')

# Load CSV data
import pandas as pd
df = pd.read_csv('path/to/your/data.csv')
events = df.to_dict('records')

# Use with commentator
commentator = AICommentator()
results = commentator.batch_analyze(events)
```

## Command Line Usage

### Test the System

```bash
python main.py --test
```

This runs a quick test to verify all components are working.

### Run Demo

```bash
python main.py --demo
```

Analyzes sample data with detailed output.

### Prepare Training Data

```bash
python main.py --prepare-data
```

Generates training, validation, and test datasets from sample and synthetic data.

### Train the Model

```bash
python main.py --train --train-config config/training_config.yaml
```

Trains the multi-task learning model on prepared data.

### Evaluate the Model

```bash
python main.py --evaluate --model models/mtl_model_best.pt
```

Evaluates the trained model on test data.

## Input Data Format

Events should be in the following format (JSON):

```json
{
  "match_id": "match_001",
  "timestamp": "45:30",
  "action_type": "goal",
  "team": "Team A",
  "player": "Player Name",
  "description": "Description of the event",
  "commentary": "Commentary text (optional)",
  "sentiment": "positive",
  "importance_score": 9.5
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| match_id | string | Unique match identifier |
| timestamp | string | Time in match (MM:SS format) |
| action_type | string | Type of action (required) |
| team | string | Team name |
| player | string | Player name |
| description | string | Event description (required) |
| commentary | string | Commentary (optional, auto-generated if missing) |
| sentiment | string | positive/neutral/negative |
| importance_score | float | 0-10 importance score |

## Output Format

The commentator returns analysis results in this format:

```python
{
    'action_type': 'goal',           # Recognized action type
    'team': 'Team A',                # Team name
    'player': 'Player Name',         # Player name
    'commentary': 'Text...',         # Generated commentary
    'sentiment': 'positive',         # Sentiment analysis result
    'importance_score': 9.5,         # Importance score (0-10)
    'confidence': 0.95               # Model confidence (0-1)
}
```

## Advanced Usage

### Using with GPU

```python
from src.inference.commentator import AICommentator

# Use GPU (will auto-detect if available)
commentator = AICommentator(device='cuda')

# Force CPU
commentator = AICommentator(device='cpu')
```

### Load Custom Model

```python
from src.inference.commentator import AICommentator

# Load trained model
commentator = AICommentator(model_path='path/to/model.pt')
```

### Data Processing

```python
from src.data_processing.preprocessing import preprocess_text, create_samples
import pandas as pd

# Preprocess text
text = "Some event description"
cleaned_text = preprocess_text(text)

# Create samples from raw data
raw_data = [...]  # list of event dicts
df = create_samples(raw_data)
print(df.head())
```

## Performance Tips

1. **Batch Processing**: Use `batch_analyze()` for multiple events to reduce overhead
2. **GPU Acceleration**: Use GPU when available for faster inference
3. **Pre-processing**: Clean and normalize your data before analysis
4. **Model Size**: Use quantized models for faster inference on resource-limited devices

## Frequently Asked Questions

### Q: How accurate is the model?
**A**: The model provides basic analysis. For production use, consider fine-tuning with your own data.

### Q: What languages are supported?
**A**: The model uses multilingual BERT, supporting Vietnamese, English, and other languages.

### Q: Can I train my own model?
**A**: Yes! Use `python main.py --train` after preparing your data.

### Q: How long does inference take?
**A**: Single event: ~0.5-2 seconds (depending on hardware). Use batch processing for multiple events.

## Troubleshooting

### Memory Issues
If you get out-of-memory errors:
- Reduce batch size in config
- Use CPU instead of GPU
- Process events one at a time

### Slow Inference
- Enable GPU acceleration
- Use batch processing
- Reduce input sequence length

### Poor Predictions
- Fine-tune model with more training data
- Check input data quality
- Adjust model configuration

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from src.inference.commentator import AICommentator

app = Flask(__name__)
commentator = AICommentator()

@app.route('/analyze', methods=['POST'])
def analyze():
    event = request.json
    result = commentator.analyze(event)
    return jsonify(result)

if __name__ == '__main__':
    app.run()
```

### Batch Processing

```python
from src.inference.commentator import AICommentator
import pandas as pd

commentator = AICommentator()

# Load CSV
df = pd.read_csv('events.csv')
events = df.to_dict('records')

# Analyze
results = commentator.batch_analyze(events)

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('analysis_results.csv', index=False)
```

## Support and Feedback

For issues, questions, or feature requests, please:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Open a new issue with detailed information

## License

MIT License - See LICENSE file for details
