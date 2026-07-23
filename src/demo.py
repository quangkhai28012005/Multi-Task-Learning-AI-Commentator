"""Demo script for testing the AI Commentator."""

import json
from pathlib import Path
from src.inference.commentator import AICommentator
from src.data_processing.data_loader import DataLoader


def demo_basic():
    """Run basic demo with sample events."""
    print("\n" + "="*60)
    print("MULTI-TASK LEARNING AI COMMENTATOR - DEMO")
    print("="*60)
    
    # Initialize commentator
    print("\n[1/3] Initializing AI Commentator...")
    commentator = AICommentator()
    print("✓ Commentator initialized successfully!")
    
    # Load sample data
    print("\n[2/3] Loading sample match data...")
    sample_data_path = 'data/samples/sample_data.json'
    
    if not Path(sample_data_path).exists():
        print(f"✗ Sample data not found at {sample_data_path}")
        return
    
    sample_events = DataLoader.load_json(sample_data_path)
    print(f"✓ Loaded {len(sample_events)} sample events")
    
    # Analyze events
    print("\n[3/3] Analyzing events...\n")
    results = commentator.batch_analyze(sample_events)
    
    # Display results
    for i, (event, result) in enumerate(zip(sample_events, results), 1):
        print("-" * 60)
        print(f"EVENT #{i}: {event.get('match_id')} at {event.get('timestamp')}")
        print("-" * 60)
        print(f"Team:              {result['team']}")
        print(f"Player:            {result['player']}")
        print(f"Description:       {event.get('description')}")
        print(f"\nAI Analysis:")
        print(f"  Action Type:     {result['action_type']}")
        print(f"  Commentary:      {result['commentary']}")
        print(f"  Sentiment:       {result['sentiment']}")
        print(f"  Importance:      {result['importance_score']}/10")
        print(f"  Confidence:      {result['confidence']*100:.1f}%")
        print()
    
    print("\n" + "="*60)
    print(f"✓ Analysis complete! Processed {len(results)} events")
    print("="*60 + "\n")


def demo_single_event():
    """Demo with a single custom event."""
    print("\n" + "="*60)
    print("SINGLE EVENT ANALYSIS")
    print("="*60 + "\n")
    
    commentator = AICommentator()
    
    # Create a custom event
    event = {
        'action_type': 'goal',
        'team': 'Manchester United',
        'player': 'Bruno Fernandes',
        'description': 'Cầu thủ thực hiện cú sút xa chính xác từ ngoài vòng cấm, bóng bay vào góc tường thợ môn không thể cấu cứu'
    }
    
    print(f"Event Information:")
    print(f"  Team:        {event['team']}")
    print(f"  Player:      {event['player']}")
    print(f"  Description: {event['description']}\n")
    
    # Analyze
    result = commentator.analyze(event)
    
    print(f"AI Analysis Results:")
    print(f"  Action Type:     {result['action_type']}")
    print(f"  Commentary:      {result['commentary']}")
    print(f"  Sentiment:       {result['sentiment']}")
    print(f"  Importance:      {result['importance_score']}/10")
    print(f"  Confidence:      {result['confidence']*100:.1f}%")
    print("\n" + "="*60 + "\n")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo AI Commentator')
    parser.add_argument('--mode', type=str, default='batch',
                       choices=['batch', 'single'],
                       help='Demo mode: batch (multiple events) or single (one event)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'batch':
            demo_basic()
        else:
            demo_single_event()
    except Exception as e:
        print(f"\n✗ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
