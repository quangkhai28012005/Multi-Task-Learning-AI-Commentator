"""Quick test script to verify the model works."""

import sys
sys.path.insert(0, '..')

from src.inference.commentator import AICommentator
from src.data_processing.data_loader import DataLoader


def quick_test():
    """Run quick test of the commentator."""
    print("\n" + "="*70)
    print("QUICK TEST - Multi-Task Learning AI Commentator")
    print("="*70)
    
    try:
        # Test 1: Initialize commentator
        print("\n[TEST 1] Initializing AI Commentator...")
        commentator = AICommentator()
        print("✓ PASS: Commentator initialized")
        
        # Test 2: Load sample data
        print("\n[TEST 2] Loading sample data...")
        try:
            sample_data = DataLoader.load_json('data/samples/sample_data.json')
            print(f"✓ PASS: Loaded {len(sample_data)} sample events")
        except FileNotFoundError:
            print("⚠ SKIP: Sample data not found (this is OK for demo)")
            sample_data = []
        
        # Test 3: Test with predefined event
        print("\n[TEST 3] Analyzing predefined event...")
        event = {
            'action_type': 'goal',
            'team': 'Manchester United',
            'player': 'Bruno Fernandes',
            'description': 'Cầu thủ sút chính xác vào góc tường, bóng bay vào lưới'
        }
        result = commentator.analyze(event)
        print(f"✓ PASS: Event analyzed")
        print(f"  - Action: {result['action_type']}")
        print(f"  - Sentiment: {result['sentiment']}")
        print(f"  - Importance: {result['importance_score']}/10")
        
        # Test 4: Batch analysis
        if sample_data:
            print("\n[TEST 4] Batch analysis...")
            results = commentator.batch_analyze(sample_data[:3])
            print(f"✓ PASS: Analyzed {len(results)} events in batch")
        
        # Test 5: Multiple events with different sentiments
        print("\n[TEST 5] Testing different event types...")
        test_events = [
            {
                'action_type': 'goal',
                'team': 'Team A',
                'player': 'Player 1',
                'description': 'Bàn thắng tuyệt vời'
            },
            {
                'action_type': 'foul',
                'team': 'Team B',
                'player': 'Player 2',
                'description': 'Phạm lỗi kỹ thuật'
            },
            {
                'action_type': 'pass',
                'team': 'Team A',
                'player': 'Player 3',
                'description': 'Đường chuyền chính xác'
            }
        ]
        
        results = commentator.batch_analyze(test_events)
        print(f"✓ PASS: Analyzed {len(results)} different event types")
        for i, result in enumerate(results, 1):
            print(f"  Event {i}: {result['action_type']} - {result['sentiment']}")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    quick_test()
