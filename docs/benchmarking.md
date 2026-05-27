# FastFlowLM Benchmarking Feature

## Overview

The `flm bench` command allows you to benchmark the performance of FastFlowLM models on your NPU hardware. It provides detailed metrics on model throughput, latency, and efficiency across different context lengths.

## Usage

### Basic Syntax
```bash
flm bench <model_tag> [options]
```

### Examples

Benchmark the Qwen 3.5 9B model:
```bash
flm bench qwen3.5:9b
```

Benchmark with a specific power mode:
```bash
flm bench qwen3.5:9b --pmode turbo
```

Benchmark with balanced power mode:
```bash
flm bench qwen3.5:9b --pmode balanced
```

## Supported Power Modes

- `powersaver` - Lowest power consumption, lower throughput
- `balanced` - Default balanced mode
- `performance` - Higher performance, higher power consumption
- `turbo` - Maximum performance

## Output Format

The benchmark produces output similar to `llama-bench` format:

```
[FLM] =========================== Benchmark Results ===========================

| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| (prefill)                      |        N/A |        N/A | NPU        |  99 |           pp1k |              234.56 ± 10.23 |
| (decoding)                     |        N/A |        N/A | NPU        |  99 |           tg128 |               45.12 ± 2.15 |
...

Summary:
   Context Length |          TTFT (s) |  Prefill (tok/s) | Decoding (tok/s)
  -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- --------
        1k |    0.125 ± 0.003 |    234.56 ± 10.23 |     45.12 ± 2.15 |
        2k |    0.245 ± 0.005 |    220.34 ± 12.10 |     43.98 ± 2.45 |
        4k |    0.456 ± 0.008 |    198.76 ± 15.30 |     42.15 ± 2.89 |
        8k |    0.890 ± 0.015 |    185.23 ± 18.90 |     40.23 ± 3.12 |
```

## Output Metrics

### TTFT (Time To First Token)
- **Definition**: The time it takes from sending a request to receiving the first token of output
- **Measured in**: Seconds
- **Includes**: Prefill latency + overhead
- **Displayed as**: `average ± standard_deviation`

### Prefill Speed (Prefill Throughput)
- **Definition**: Number of prompt tokens processed per second during the prefill phase
- **Measured in**: Tokens per second (tok/s)
- **Higher is better**: More efficient prompt processing
- **Displayed as**: `average ± standard_deviation`

### Decoding Speed (Autoregressive Generation Speed)
- **Definition**: Number of output tokens generated per second during the decoding phase
- **Measured in**: Tokens per second (tok/s)
- **Higher is better**: Faster text generation
- **Displayed as**: `average ± standard_deviation`

## Benchmark Configuration

By default, the benchmark:
- Tests context lengths: 1k, 2k, 4k, 8k, 16k, 32k (up to max_length in config)
- Runs **5 iterations** per context length
- Uses a comprehensive input text (~20k tokens)
- Generates 32 tokens per test

To customize the benchmark, create or modify `bench_config.json`:

```json
{
    "input_text": "Your benchmark prompt text here...",
    "iterations": 8,
    "max_length": 32768
}
```

Then run:
```bash
flm bench qwen3.5:9b --prompt bench_config.json
```

## Output Files

Benchmark results are automatically saved to CSV files in the current directory:
```
bench_qwen35_9b_20260527_[cpu_info].csv
```

This CSV includes detailed statistics for:
- Context length in 1k increments
- TTFT (time, std dev, min, max)
- Prefill speed (tokens/s, std dev, min, max)
- Decoding speed (tokens/s, std dev, min, max)

## Performance Analysis Tips

### Expected Performance Ranges

**For TTFT:**
- Good: < 1 second
- Excellent: < 0.5 seconds

**For Prefill Speed:**
- Good: > 100 tok/s
- Excellent: > 200 tok/s

**For Decoding Speed (Generation):**
- Good: > 20 tok/s
- Excellent: > 40 tok/s

### Factors Affecting Performance

1. **Context Length**: Performance typically decreases with larger context windows
2. **Power Mode**: Higher power modes provide better throughput
3. **Model Size**: Larger models are generally slower
4. **NPU Load**: Other processes using the NPU will impact results
5. **Kernel**: Specific context lengths may perform differently

## Interpreting Results

### High Standard Deviation
- Indicates **inconsistent performance**, possibly due to:
  - Thermal throttling
  - OS scheduling variations
  - Other system processes interfering
  - **Solution**: Run during quiet system periods or increase iterations

### Performance Decrease with Context Length
- **Normal**: Larger contexts require more computation
- **Expected**: 10-30% drop per doubling of context length
- **Concern**: > 50% drop may indicate memory bandwidth limitations

### Comparison Between Prefill and Decoding
- **Prefill-heavy**: Large batch processing efficiency (better prefill speed)
- **Decoding-heavy**: Real-time generation efficiency (better decoding speed)
- **Balanced**: Good for both interactive and batch workloads

## Example: Comparing Models

Benchmark multiple models for comparison:

```bash
# Benchmark Qwen 3.5 9B
flm bench qwen3.5:9b --pmode performance

# Benchmark Llama 3.2 3B
flm bench llama3.2:3b --pmode performance

# Benchmark Gemma 3 4B
flm bench gemma3:4b --pmode performance
```

Then compare the CSV output files to identify which model offers the best throughput for your use case.

## Benchmarking Best Practices

1. **Close other applications** to reduce system noise
2. **Run multiple iterations** (8+) to get stable statistics
3. **Use consistent power modes** across benchmarks for fair comparison
4. **Allow system to stabilize** - run a warm-up benchmark first
5. **Check thermal conditions** - excessive throttling invalidates results
6. **Test at different context lengths** - model behavior varies significantly

## Power Mode Recommendations

- **Development/Testing**: Use `balanced` for reproducible results
- **Production/Heavy Load**: Use `performance` or `turbo` (monitor thermals)
- **Battery/Low-Power**: Use `powersaver` (expect reduced throughput)
- **Comparative Benchmarks**: Use same mode for all runs

## Integration with CI/CD

The benchmark results are saved as CSV files, making them suitable for:
- Performance regression tracking
- Automated performance dashboards
- Historical trend analysis
- Model selection automation

## Advanced Usage

### Custom Input Text
Create a file with your benchmark prompt:
```bash
flm bench qwen3.5:9b --prompt my_prompt.json
```

Format of `my_prompt.json`:
```json
{
    "input_text": "Your custom prompt...",
    "iterations": 10,
    "max_length": 16384
}
```

### Power Mode Control
```bash
# Maximum performance (highest power consumption)
flm bench qwen3.5:9b --pmode turbo

# Balanced mode (recommended for comparing models)
flm bench qwen3.5:9b --pmode balanced
```

## Troubleshooting

### Error: Model not found
```bash
# Use correct model tag
flm list  # See available models
```

### Error: NPU device not found
```bash
# Validate NPU stack
flm validate

# Check NPU connectivity and drivers
```

### Low throughput numbers
- Check if other processes are using the NPU
- Verify thermal conditions
- Try different power modes
- Ensure model is properly loaded

### Inconsistent results
- Increase iterations to get better statistics
- Close other applications
- Check system thermal conditions
- Verify power mode consistency

## Technical Details

### Benchmark Stages
The benchmark runs tests across multiple context lengths (stages):
- Stage 0: 1k tokens
- Stage 1: 2k tokens
- Stage 2: 4k tokens
- Stage 3: 8k tokens
- Stage 4: 16k tokens
- Stage 5: 32k tokens (if max_length allows)

Each stage is tested in reverse order (largest first) to catch memory errors early.

### Statistics Calculation
For each metric:
- **Average**: Mean of all iterations
- **Standard Deviation**: Measure of result variability
- **Min/Max**: Range of observed values

### CSV Format
The CSV output includes columns for each context length:
```
context_length_k,
ttft_avg_s,ttft_std_s,ttft_min_s,ttft_max_s,
prefill_avg_toks_per_s,prefill_std_toks_per_s,prefill_min_toks_per_s,prefill_max_toks_per_s,
decoding_avg_toks_per_s,decoding_std_toks_per_s,decoding_min_toks_per_s,decoding_max_toks_per_s
```

## See Also

- [FastFlowLM Documentation](README.md)
- [`flm run`](README.md) - Interactive model execution
- [`flm serve`](README.md) - Start a server
- [`flm list`](README.md) - List available models

## Contributing

Found an issue with benchmarking? Please report it with:
1. Model tested
2. Power mode used
3. Output CSV file
4. System information (run `flm validate`)
