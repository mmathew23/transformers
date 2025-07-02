#!/usr/bin/env python3
"""
Test script to demonstrate generation timing functionality.

This script shows how to use the timing utilities to measure generation performance
across different models and generation modes.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import get_generation_timing_stats, clear_generation_timing_stats

def test_generation_timing():
    """Test the generation timing functionality with a simple model."""
    
    # Clear any existing stats
    clear_generation_timing_stats()
    
    # Load a small model for testing
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test input
    input_text = "Hello, how are you?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    print(f"Testing generation timing with model: {model_name}")
    print(f"Input text: {input_text}")
    print(f"Input length: {input_ids.shape[1]}")
    print("-" * 50)
    
    # Test different generation modes
    generation_configs = [
        {"do_sample": False, "max_new_tokens": 10, "name": "Greedy"},
        {"do_sample": True, "temperature": 0.7, "max_new_tokens": 10, "name": "Sampling"},
        {"num_beams": 3, "max_new_tokens": 10, "name": "Beam Search"},
    ]
    
    for config in generation_configs:
        print(f"\nTesting {config['name']} generation:")
        
        # Clear stats before each test
        clear_generation_timing_stats()
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                **{k: v for k, v in config.items() if k != 'name'},
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Get timing stats
        stats = get_generation_timing_stats()
        
        if stats:
            stat = stats[0]  # Get the first (and only) stat
            print(f"  Model: {stat.model_name}")
            print(f"  Call depth: {stat.call_depth}")
            print(f"  Total time: {stat.total_time:.4f}s")
            print(f"  Setup time: {stat.setup_time:.4f}s")
            print(f"  Sample time: {stat.sample_time:.4f}s")
            print(f"  Remainder time: {stat.remainder_time:.4f}s")
            print(f"  Cache type: {stat.cache_type}")
            print(f"  Cache size: {stat.cache_size}")
            print(f"  Generation mode: {stat.generation_mode}")
            print(f"  Batch size: {stat.batch_size}")
            print(f"  Input length: {stat.input_length}")
            print(f"  Output length: {stat.output_length}")
            
            # Decode and show output
            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            print(f"  Generated text: {generated_text}")
        else:
            print("  No timing stats available")
    
    print("\n" + "=" * 50)
    print("All timing stats:")
    all_stats = get_generation_timing_stats()
    for i, stat in enumerate(all_stats):
        print(f"\nStat {i+1}:")
        print(f"  Model: {stat.model_name}")
        print(f"  Total time: {stat.total_time:.4f}s")
        print(f"  Setup time: {stat.setup_time:.4f}s")
        print(f"  Sample time: {stat.sample_time:.4f}s")
        print(f"  Generation mode: {stat.generation_mode}")

def test_nested_generation():
    """Test nested generation calls (simulating a custom sampler scenario)."""
    
    print("\n" + "=" * 50)
    print("Testing nested generation calls...")
    
    # Clear stats
    clear_generation_timing_stats()
    
    # Load model
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_text = "The weather is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    print(f"Input: {input_text}")
    
    # Simulate a custom sampler that calls generate on a submodel
    class CustomSampler:
        def __init__(self, main_model, sub_model):
            self.main_model = main_model
            self.sub_model = sub_model
        
        def sample(self, input_ids):
            # First call: main model
            print("  Calling main model generate...")
            main_output = self.main_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Second call: sub model (nested)
            print("  Calling sub model generate...")
            sub_input = torch.cat([input_ids, main_output.sequences[:, -1:]], dim=-1)
            sub_output = self.sub_model.generate(
                sub_input,
                max_new_tokens=3,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return main_output, sub_output
    
    # Create custom sampler with same model (for demonstration)
    sampler = CustomSampler(model, model)
    
    # Run nested generation
    with torch.no_grad():
        main_output, sub_output = sampler.sample(input_ids)
    
    # Get all timing stats
    stats = get_generation_timing_stats()
    print(f"\nCollected {len(stats)} timing stats:")
    
    for i, stat in enumerate(stats):
        print(f"\nCall {i+1}:")
        print(f"  Model: {stat.model_name}")
        print(f"  Call depth: {stat.call_depth}")
        print(f"  Total time: {stat.total_time:.4f}s")
        print(f"  Setup time: {stat.setup_time:.4f}s")
        print(f"  Sample time: {stat.sample_time:.4f}s")
        print(f"  Generation mode: {stat.generation_mode}")
        print(f"  Input length: {stat.input_length}")
        print(f"  Output length: {stat.output_length}")

if __name__ == "__main__":
    print("Generation Timing Test")
    print("=" * 50)
    
    try:
        test_generation_timing()
        test_nested_generation()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 