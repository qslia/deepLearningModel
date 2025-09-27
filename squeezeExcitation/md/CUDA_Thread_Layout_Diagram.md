# CUDA Thread Layout and Indexing Pattern

## Overview
This diagram explains how `int idx = blockIdx.x * blockDim.x + threadIdx.x;` works in CUDA programming.

## Visual Layout

### Grid Structure
```
CUDA Grid (Multiple Blocks)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Block 0           Block 1           Block 2           Block 3  │
│ ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐│
│ │Thread 0 │       │Thread 0 │       │Thread 0 │       │Thread 0 ││
│ │Thread 1 │       │Thread 1 │       │Thread 1 │       │Thread 1 ││
│ │Thread 2 │       │Thread 2 │       │Thread 2 │       │Thread 2 ││
│ │   ...   │       │   ...   │       │   ...   │       │   ...   ││
│ │Thread255│       │Thread255│       │Thread255│       │Thread255││
│ └─────────┘       └─────────┘       └─────────┘       └─────────┘│
│                                                                 │
│ blockIdx.x = 0    blockIdx.x = 1    blockIdx.x = 2    blockIdx.x = 3
└─────────────────────────────────────────────────────────────────┘
```

### Thread Indexing Formula
```
Global Thread ID = blockIdx.x × blockDim.x + threadIdx.x
```

### Example Calculation (blockDim.x = 256)

#### Block 0:
```
┌─────────────────┬──────────────────┬─────────────────────┐
│ threadIdx.x     │ Calculation      │ Global Thread ID    │
├─────────────────┼──────────────────┼─────────────────────┤
│ 0               │ 0×256 + 0        │ 0                   │
│ 1               │ 0×256 + 1        │ 1                   │
│ 2               │ 0×256 + 2        │ 2                   │
│ ...             │ ...              │ ...                 │
│ 255             │ 0×256 + 255      │ 255                 │
└─────────────────┴──────────────────┴─────────────────────┘
```

#### Block 1:
```
┌─────────────────┬──────────────────┬─────────────────────┐
│ threadIdx.x     │ Calculation      │ Global Thread ID    │
├─────────────────┼──────────────────┼─────────────────────┤
│ 0               │ 1×256 + 0        │ 256                 │
│ 1               │ 1×256 + 1        │ 257                 │
│ 2               │ 1×256 + 2        │ 258                 │
│ ...             │ ...              │ ...                 │
│ 255             │ 1×256 + 255      │ 511                 │
└─────────────────┴──────────────────┴─────────────────────┘
```

#### Block 2:
```
┌─────────────────┬──────────────────┬─────────────────────┐
│ threadIdx.x     │ Calculation      │ Global Thread ID    │
├─────────────────┼──────────────────┼─────────────────────┤
│ 0               │ 2×256 + 0        │ 512                 │
│ 1               │ 2×256 + 1        │ 513                 │
│ 2               │ 2×256 + 2        │ 514                 │
│ ...             │ ...              │ ...                 │
│ 255             │ 2×256 + 255      │ 767                 │
└─────────────────┴──────────────────┴─────────────────────┘
```

## Memory Layout Mapping

### Example: Global Average Pooling
For a tensor with shape `[batch_size=2, channels=4, height=8, width=8]`:

```
Data Layout in Memory:
┌─────────────────────────────────────────────────────────────┐
│ Batch 0, Channel 0 │ Batch 0, Channel 1 │ ... │ Batch 1, Channel 3 │
│     (64 elements)   │     (64 elements)   │ ... │     (64 elements)   │
└─────────────────────────────────────────────────────────────┘

Thread Mapping:
┌────────────┬─────────────┬──────────────┬─────────────────────┐
│ Global ID  │ Batch Index │ Channel Index│ Processes           │
├────────────┼─────────────┼──────────────┼─────────────────────┤
│ 0          │ 0           │ 0            │ Batch 0, Channel 0  │
│ 1          │ 0           │ 1            │ Batch 0, Channel 1  │
│ 2          │ 0           │ 2            │ Batch 0, Channel 2  │
│ 3          │ 0           │ 3            │ Batch 0, Channel 3  │
│ 4          │ 1           │ 0            │ Batch 1, Channel 0  │
│ 5          │ 1           │ 1            │ Batch 1, Channel 1  │
│ 6          │ 1           │ 2            │ Batch 1, Channel 2  │
│ 7          │ 1           │ 3            │ Batch 1, Channel 3  │
└────────────┴─────────────┴──────────────┴─────────────────────┘
```

## Code Context

### From the CUDA kernel:
```cuda
__global__ void global_avg_pool_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // ← This line!
    int total_elements = batch_size * channels;
    
    if (idx < total_elements) {
        int batch_idx = idx / channels;      // Which batch
        int channel_idx = idx % channels;    // Which channel
        
        // Each thread processes one (batch, channel) combination
        // and computes average across spatial dimensions (height × width)
    }
}
```

### Host code setup:
```cuda
const int threads = 256;  // blockDim.x
const int blocks = (batch_size * channels + threads - 1) / threads;  // gridDim.x

global_avg_pool_forward_kernel<<<blocks, threads>>>(
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    batch_size, channels, height, width);
```

## Key Benefits

1. **Unique Identification**: Each thread gets a unique global ID
2. **Parallel Processing**: Multiple threads work simultaneously
3. **Memory Coalescing**: Sequential threads access sequential memory locations
4. **Scalability**: Works with any number of blocks and threads
5. **Load Balancing**: Work is evenly distributed across all threads

## Common Pattern Usage

This indexing pattern appears in all CUDA kernels in the squeeze excitation implementation:
- Global Average Pooling (forward & backward)
- Excitation operations (forward & backward) 
- Activation functions (Swish, Sigmoid)

Each kernel uses the same fundamental approach to map threads to data elements for efficient parallel processing.
