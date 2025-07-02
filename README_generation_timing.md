# Generation Timing Utilities

This document describes the timing utilities added to the `transformers` library to measure generation performance across different models and generation modes.

## Overview

The timing utilities allow you to measure:
- **Total generation time**: Complete time from start to finish
- **Setup time**: Time spent in initialization and preparation phases
- **Sample time**: Time spent in the actual generation/sampling phase
- **Remainder time**: Time spent in post-processing and cleanup
- **Sample-specific timing**: Detailed breakdown of the `_sample` method
  - **Sample total time**: Total time spent in the `_sample` method
  - **Sample forward time**: Time spent in model forward passes within sampling
  - **Sample sampling time**: Time spent in token selection and sampling logic
- **Cache information**: Type, size, and reset status of the model cache
- **Generation metadata**: Mode, batch size, input/output lengths

The system supports nested generation calls, allowing you to differentiate between calls from different models (e.g., main model vs. submodel in a custom sampler).

## Key Components

### 1. GenerationTimingStats Dataclass

Contains all timing and metadata information:

```python
@dataclass
class GenerationTimingStats:
    model_name: str
    call_depth: int
    total_time: float
    setup_time: float
    sample_time: float
    remainder_time: float
    # Sample-specific timing
    sample_total_time: float = 0.0
    sample_forward_time: float = 0.0
    sample_sampling_time: float = 0.0
    cache_type: str = ""
    cache_size: Optional[int] = None
    cache_reset: bool = False
    generation_mode: str = ""
    batch_size: int = 0
    input_length: int = 0
    output_length: int = 0
```

### 2. GenerationTimingContext Class

Manages timing across nested calls with methods for:
- `start_call()` / `end_call()`: Track generation calls
- `mark_setup_start()` / `mark_sample_start()` / `mark_sample_end()`: Phase timing
- `mark_sample_forward_start()` / `mark_sample_forward_end()`: Forward pass timing
- `mark_sample_sampling_start()` / `mark_sample_sampling_end()`: Sampling timing
- `update_cache_info()`: Cache metadata
- `update_generation_info()`: Generation parameters

### 3. Global Functions

- `get_generation_timing_stats()`: Retrieve all collected statistics
- `clear_generation_timing_stats()`: Clear collected statistics
- `generation_timing_context()`: Context manager for timing

## Usage Examples

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import get_generation_timing_stats, clear_generation_timing_stats

# Clear any existing stats
clear_generation_timing_stats()

# Load model and generate
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt")
outputs = model.generate(input_ids, max_new_tokens=20, do_sample=True)

# Get timing statistics
stats = get_generation_timing_stats()
for stat in stats:
    print(f"Model: {stat.model_name}")
    print(f"Total time: {stat.total_time:.4f}s")
    print(f"Sample time: {stat.sample_time:.4f}s")
    print(f"Sample forward time: {stat.sample_forward_time:.4f}s")
    print(f"Sample sampling time: {stat.sample_sampling_time:.4f}s")
```

### Nested Generation Calls

The system automatically tracks nested calls with different call depths:

```python
# First generation call (depth 0)
outputs1 = model.generate(input_ids, max_new_tokens=10)

# Second generation call (depth 0) - appears as separate call
outputs2 = model.generate(input_ids, max_new_tokens=5)

# If you have a custom sampler that calls generate internally,
# those calls would appear with depth > 0
```

### Cache Information

Cache details are automatically collected:

```python
stats = get_generation_timing_stats()
for stat in stats:
    print(f"Cache type: {stat.cache_type}")
    print(f"Cache size: {stat.cache_size}")
    print(f"Cache reset: {stat.cache_reset}")
```

## Timing Breakdown

### Generation Phases

1. **Setup Phase**: Initialization, configuration, input preparation
2. **Sample Phase**: The main generation loop in `_sample` method
3. **Remainder Phase**: Post-processing, output formatting

### Sample-Specific Phases

Within the `_sample` method:

1. **Forward Pass**: Model inference (`model.forward()` calls)
2. **Sampling**: Token selection (softmax, multinomial/argmax)
3. **Other**: Loop overhead, stopping criteria, etc.

## Implementation Details

### Integration Points

The timing is integrated into:

1. **`generate()` method**: Main entry point with setup/sample/remainder phases
2. **`_sample()` method**: Detailed forward and sampling timing
3. **Cache management**: Automatic cache information collection

### Performance Impact

The timing overhead is minimal:
- Time measurements use `time.time()` (microsecond precision)
- No additional memory allocations during timing
- Timing data is collected in a lightweight context manager

### Thread Safety

The global timing context is not thread-safe. For multi-threaded applications, consider:
- Using separate timing contexts per thread
- Collecting timing data in a thread-local storage
- Synchronizing access to the global context

## Advanced Usage

### Custom Timing Context

```python
from transformers.generation.utils import generation_timing_context

with generation_timing_context() as timing_ctx:
    # Your generation code here
    outputs = model.generate(input_ids, max_new_tokens=20)
    
    # Access timing data directly
    stats = timing_ctx.get_stats()
```

### Filtering and Analysis

```python
stats = get_generation_timing_stats()

# Filter by model
gpt_stats = [s for s in stats if "GPT" in s.model_name]

# Filter by generation mode
sampling_stats = [s for s in stats if s.generation_mode == "sampling"]

# Analyze performance
avg_total_time = sum(s.total_time for s in stats) / len(stats)
avg_forward_time = sum(s.sample_forward_time for s in stats) / len(stats)
```

## Troubleshooting

### No Timing Data

If no timing data is collected:
1. Ensure you're using the patched `transformers` version
2. Check that `generate()` is called (not direct model inference)
3. Verify the timing context is properly initialized

### Inconsistent Results

For consistent timing:
1. Warm up the model before timing measurements
2. Use consistent input sizes and generation parameters
3. Consider GPU synchronization for CUDA operations
4. Account for JIT compilation overhead in first runs

### Memory Usage

The timing system stores minimal data:
- One `GenerationTimingStats` object per generation call
- Call stack depth information
- No tensor or model data is stored

## Future Enhancements

Potential improvements:
- GPU timing integration (CUDA events)
- Memory usage tracking
- Per-layer timing breakdown
- Export to profiling tools (TensorBoard, etc.)
- Async timing support
- Distributed generation timing 