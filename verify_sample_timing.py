#!/usr/bin/env python3
"""
Verification script for sample-specific timing implementation.

This script checks that the sample-specific timing utilities are properly implemented
and can be imported without errors.
"""

import sys
import os

def verify_sample_timing_implementation():
    """Verify that the sample-specific timing implementation is syntactically correct."""
    
    # Add the src directory to the path so we can import from transformers
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        # Try to import the timing utilities
        from transformers.generation.utils import (
            GenerationTimingStats,
            GenerationTimingContext,
            get_generation_timing_stats,
            clear_generation_timing_stats,
            _generation_timing_context
        )
        
        print("✅ Successfully imported timing utilities")
        
        # Check that GenerationTimingStats has the new sample-specific fields
        stats = GenerationTimingStats(
            model_name="TestModel",
            call_depth=0,
            total_time=1.0,
            setup_time=0.1,
            sample_time=0.8,
            remainder_time=0.1,
            sample_total_time=0.8,
            sample_forward_time=0.6,
            sample_sampling_time=0.2,
            cache_type="DynamicCache",
            cache_size=100,
            cache_reset=False,
            generation_mode="SAMPLE",
            batch_size=1,
            input_length=10,
            output_length=20
        )
        
        print("✅ GenerationTimingStats supports sample-specific timing fields")
        print(f"   Sample total time: {stats.sample_total_time}")
        print(f"   Sample forward time: {stats.sample_forward_time}")
        print(f"   Sample sampling time: {stats.sample_sampling_time}")
        
        # Check that GenerationTimingContext has the new methods
        context = GenerationTimingContext()
        
        # Test sample-specific timing methods
        context.start_call("TestModel")
        context.mark_sample_forward_start()
        context.mark_sample_forward_end()
        context.mark_sample_sampling_start()
        context.mark_sample_sampling_end()
        
        print("✅ GenerationTimingContext supports sample-specific timing methods")
        
        # Test global context
        clear_generation_timing_stats()
        stats_list = get_generation_timing_stats()
        print(f"✅ Global timing context works (cleared, got {len(stats_list)} stats)")
        
        print("\n🎉 All sample-specific timing functionality verified successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def check_sample_method_integration():
    """Check that the _sample method has timing integration."""
    
    try:
        # Read the _sample method to check for timing integration
        sample_file = os.path.join('src', 'transformers', 'generation', 'utils.py')
        
        if not os.path.exists(sample_file):
            print(f"❌ Could not find {sample_file}")
            return False
        
        with open(sample_file, 'r') as f:
            content = f.read()
        
        # Check for timing integration in _sample method
        timing_checks = [
            "sample_start_time = time.time()",
            "_generation_timing_context.mark_sample_forward_start()",
            "_generation_timing_context.mark_sample_forward_end()",
            "_generation_timing_context.mark_sample_sampling_start()",
            "_generation_timing_context.mark_sample_sampling_end()",
            "sample_total_time = time.time() - sample_start_time"
        ]
        
        missing_checks = []
        for check in timing_checks:
            if check not in content:
                missing_checks.append(check)
        
        if missing_checks:
            print(f"❌ Missing timing integration in _sample method:")
            for check in missing_checks:
                print(f"   - {check}")
            return False
        else:
            print("✅ _sample method has complete timing integration")
            return True
            
    except Exception as e:
        print(f"❌ Error checking _sample method integration: {e}")
        return False

if __name__ == "__main__":
    print("Verifying sample-specific timing implementation...")
    print("=" * 50)
    
    success1 = verify_sample_timing_implementation()
    success2 = check_sample_method_integration()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All verifications passed! Sample-specific timing is ready to use.")
        sys.exit(0)
    else:
        print("❌ Some verifications failed. Please check the implementation.")
        sys.exit(1) 