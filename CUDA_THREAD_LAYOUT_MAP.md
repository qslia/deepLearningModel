# CUDA Thread Layout and Indexing Map

## Overview
This document visualizes how CUDA threads are organized and indexed using the formula:
```
idx = blockIdx.x * blockDim.x + threadIdx.x
```

## Thread Organization Hierarchy

```
GPU Grid
├── Block 0
│   ├── Thread 0 (threadIdx.x = 0)
│   ├── Thread 1 (threadIdx.x = 1)
│   ├── Thread 2 (threadIdx.x = 2)
│   └── ... (up to blockDim.x - 1)
├── Block 1
│   ├── Thread 0 (threadIdx.x = 0)
│   ├── Thread 1 (threadIdx.x = 1)
│   └── ...
└── Block N
    └── ...
```

## Visual Thread Layout Example

**Example Configuration:**
- `blockDim.x = 4` (4 threads per block)
- `gridDim.x = 3` (3 blocks total)

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU GRID                            │
├─────────────────────────────────────────────────────────────┤
│  BLOCK 0        │  BLOCK 1        │  BLOCK 2              │
│  (blockIdx=0)   │  (blockIdx=1)   │  (blockIdx=2)         │
├─────────────────┼─────────────────┼─────────────────────────┤
│ Thread 0        │ Thread 0        │ Thread 0              │
│ threadIdx.x=0   │ threadIdx.x=0   │ threadIdx.x=0         │
│ Global idx=0    │ Global idx=4    │ Global idx=8          │
│                 │                 │                       │
│ Thread 1        │ Thread 1        │ Thread 1              │
│ threadIdx.x=1   │ threadIdx.x=1   │ threadIdx.x=1         │
│ Global idx=1    │ Global idx=5    │ Global idx=9          │
│                 │                 │                       │
│ Thread 2        │ Thread 2        │ Thread 2              │
│ threadIdx.x=2   │ threadIdx.x=2   │ threadIdx.x=2         │
│ Global idx=2    │ Global idx=6    │ Global idx=10         │
│                 │                 │                       │
│ Thread 3        │ Thread 3        │ Thread 3              │
│ threadIdx.x=3   │ threadIdx.x=3   │ threadIdx.x=3         │
│ Global idx=3    │ Global idx=7    │ Global idx=11         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Index Calculation Table

| Block ID | Thread ID | Calculation | Global Index |
|----------|-----------|-------------|--------------|
| 0        | 0         | 0×4 + 0     | 0            |
| 0        | 1         | 0×4 + 1     | 1            |
| 0        | 2         | 0×4 + 2     | 2            |
| 0        | 3         | 0×4 + 3     | 3            |
| 1        | 0         | 1×4 + 0     | 4            |
| 1        | 1         | 1×4 + 1     | 5            |
| 1        | 2         | 1×4 + 2     | 6            |
| 1        | 3         | 1×4 + 3     | 7            |
| 2        | 0         | 2×4 + 0     | 8            |
| 2        | 1         | 2×4 + 1     | 9            |
| 2        | 2         | 2×4 + 2     | 10           |
| 2        | 3         | 2×4 + 3     | 11           |

## Real-World Example: Global Average Pooling

In the `global_avg_pool_forward_kernel` from your code:

**Input Tensor Dimensions:**
- Batch size: 2
- Channels: 8
- Height: 32
- Width: 32

**Thread Assignment:**
```
┌─────────────────────────────────────────────────────────────┐
│               Data Layout (Batch × Channel)                │
├─────────────────────────────────────────────────────────────┤
│  Batch 0                    │  Batch 1                    │
├─────────────────────────────┼─────────────────────────────┤
│ Ch0  Ch1  Ch2  Ch3  Ch4  Ch5│ Ch0  Ch1  Ch2  Ch3  Ch4  Ch5│
│ Ch6  Ch7                    │ Ch6  Ch7                    │
├─────────────────────────────┼─────────────────────────────┤
│ idx0 idx1 idx2 idx3 idx4 idx5│ idx8 idx9 idx10 idx11 idx12 idx13│
│ idx6 idx7                   │ idx14 idx15                 │
└─────────────────────────────┴─────────────────────────────┘
```

**Thread-to-Data Mapping:**
- Thread with `idx = 0` → Batch 0, Channel 0
- Thread with `idx = 1` → Batch 0, Channel 1
- Thread with `idx = 8` → Batch 1, Channel 0
- Thread with `idx = 15` → Batch 1, Channel 7

Each thread computes the average of all spatial locations (32×32=1024 pixels) for its assigned batch-channel combination.

## Memory Access Pattern

```
Input Tensor Memory Layout (flattened):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ B0C0│ B0C1│ B0C2│ ... │ B0C7│ B1C0│ B1C1│ ... │
│ All │ All │ All │     │ All │ All │ All │     │
│ HxW │ HxW │ HxW │     │ HxW │ HxW │ HxW │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑     ↑     ↑           ↑     ↑     ↑
Thread Thread Thread   Thread Thread Thread
idx=0  idx=1  idx=2    idx=7  idx=8  idx=9
```

## Key Benefits of This Indexing

1. **Unique Assignment**: Each thread gets exactly one unique piece of work
2. **Coalesced Memory Access**: Adjacent threads access adjacent memory locations
3. **Load Balancing**: Work is evenly distributed across all threads
4. **Scalability**: Pattern works regardless of problem size

## Formula Variations

### 2D Grid (for 2D problems):
```c
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = idx_y * width + idx_x;  // Convert to 1D index
```

### With Bounds Checking:
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < total_elements) {
    // Process element at index idx
}
```

This indexing pattern is fundamental to CUDA programming and ensures efficient parallel execution across thousands of GPU cores!
