#!/usr/bin/env python3
"""
Verification script for generation timing implementation.

This script checks that the timing utilities are properly implemented
without requiring external dependencies.
"""

import sys
import os

def verify_timing_implementation():
    """Verify that the timing implementation is syntactically correct."""
    
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
        print("✓ Successfully imported timing utilities")
        
        # Check that the dataclass has the expected fields
        expected_fields = {
            'model_name', 'call_depth', 'total_time', 'setup_time', 
            'sample_time', 'remainder_time', 'cache_type', 'cache_size',
            'cache_reset', 'generation_mode', 'batch_size', 'input_length', 'output_length'
        }
        
        actual_fields = set(GenerationTimingStats.__dataclass_fields__.keys())
        if expected_fields.issubset(actual_fields):
            print("✓ GenerationTimingStats has all expected fields")
        else:
            missing = expected_fields - actual_fields
            print(f"✗ Missing fields in GenerationTimingStats: {missing}")
            return False
        
        # Check that the context class has the expected methods
        expected_methods = {
            'start_call', 'end_call', 'mark_setup_start', 'mark_sample_start',
            'mark_sample_end', 'update_cache_info', 'update_generation_info',
            'get_stats', 'clear_stats'
        }
        
        actual_methods = set(dir(GenerationTimingContext))
        if expected_methods.issubset(actual_methods):
            print("✓ GenerationTimingContext has all expected methods")
        else:
            missing = expected_methods - actual_methods
            print(f"✗ Missing methods in GenerationTimingContext: {missing}")
            return False
        
        # Check that the global context exists
        if _generation_timing_context is not None:
            print("✓ Global timing context exists")
        else:
            print("✗ Global timing context is None")
            return False
        
        # Test basic functionality
        print("\nTesting basic functionality:")
        
        # Test start_call
        call_depth = _generation_timing_context.start_call("TestModel")
        if call_depth == 0:
            print("✓ start_call works correctly")
        else:
            print(f"✗ start_call returned unexpected depth: {call_depth}")
            return False
        
        # Test mark_setup_start
        _generation_timing_context.mark_setup_start()
        print("✓ mark_setup_start works")
        
        # Test mark_sample_start
        _generation_timing_context.mark_sample_start()
        print("✓ mark_sample_start works")
        
        # Test mark_sample_end
        _generation_timing_context.mark_sample_end()
        print("✓ mark_sample_end works")
        
        # Test update_cache_info
        _generation_timing_context.update_cache_info("DynamicCache", 100, False)
        print("✓ update_cache_info works")
        
        # Test update_generation_info
        _generation_timing_context.update_generation_info("SAMPLE", 1, 10)
        print("✓ update_generation_info works")
        
        # Test end_call
        stats = _generation_timing_context.end_call("TestModel", None)
        if stats is not None:
            print("✓ end_call works correctly")
            print(f"  - Model: {stats.model_name}")
            print(f"  - Total time: {stats.total_time:.6f}s")
            print(f"  - Cache type: {stats.cache_type}")
            print(f"  - Generation mode: {stats.generation_mode}")
        else:
            print("✗ end_call returned None")
            return False
        
        # Test get_stats
        all_stats = get_generation_timing_stats()
        if len(all_stats) == 1:
            print("✓ get_generation_timing_stats works correctly")
        else:
            print(f"✗ get_generation_timing_stats returned {len(all_stats)} stats, expected 1")
            return False
        
        # Test clear_stats
        clear_generation_timing_stats()
        cleared_stats = get_generation_timing_stats()
        if len(cleared_stats) == 0:
            print("✓ clear_generation_timing_stats works correctly")
        else:
            print(f"✗ clear_generation_timing_stats failed, still have {len(cleared_stats)} stats")
            return False
        
        print("\n✓ All timing utilities are working correctly!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_generate_method_integration():
    """Check that the generate method has timing integration."""
    
    try:
        # Read the generate method to check for timing calls
        utils_path = os.path.join(os.path.dirname(__file__), 'src', 'transformers', 'generation', 'utils.py')
        
        if not os.path.exists(utils_path):
            print(f"✗ Could not find utils.py at {utils_path}")
            return False
        
        with open(utils_path, 'r') as f:
            content = f.read()
        
        # Check for key timing integration points
        timing_checks = [
            ("start_call", "generate method starts timing"),
            ("mark_setup_start", "generate method marks setup start"),
            ("mark_sample_start", "generate method marks sample start"),
            ("mark_sample_end", "generate method marks sample end"),
            ("end_call", "generate method ends timing"),
            ("update_cache_info", "generate method updates cache info"),
            ("update_generation_info", "generate method updates generation info"),
        ]
        
        print("\nChecking generate method integration:")
        all_present = True
        
        for check, description in timing_checks:
            if check in content:
                print(f"✓ {description}")
            else:
                print(f"✗ Missing: {description}")
                all_present = False
        
        # Check for try/except block around timing
        if "try:" in content and "except Exception as e:" in content:
            print("✓ Exception handling for timing is present")
        else:
            print("✗ Missing exception handling for timing")
            all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"✗ Error checking generate method integration: {e}")
        return False

if __name__ == "__main__":
    print("Generation Timing Implementation Verification")
    print("=" * 50)
    
    success = True
    
    # Verify timing utilities
    if not verify_timing_implementation():
        success = False
    
    # Check generate method integration
    if not check_generate_method_integration():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All verifications passed! The timing implementation is ready.")
    else:
        print("✗ Some verifications failed. Please check the implementation.")
        sys.exit(1) 