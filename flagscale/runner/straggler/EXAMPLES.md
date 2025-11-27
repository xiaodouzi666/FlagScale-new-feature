# Straggler Detection Module - Usage Examples

## Basic Usage

### 1. Configure and Initialize the Detector

```python
from flagscale.runner.straggler import StragglerConfig, StragglerDetector

# Create configuration
config = StragglerConfig(
    enabled=True,
    scores_to_compute="all",
    profiling_interval=10,
    report_interval_steps=100,
    monitor_sections=["dataloader", "forward", "backward", "optimizer"],
    straggler_threshold=1.5,
)

# Initialize detector
detector = StragglerDetector(
    config=config,
    rank=0,
    world_size=4,
    node_name="node-0",
)
```

### 2. Profile Training Sections

```python
# Option 1: Use context manager
with SectionContext(detector, "forward", profile_cuda=True):
    output = model(input)

with SectionContext(detector, "backward", profile_cuda=True):
    loss.backward()

# Option 2: Use decorator
@create_section_decorator(detector, "data_loading")
def load_batch():
    return dataloader.next_batch()

# Option 3: Manual recording
detector.record_section("optimizer", cpu_time=0.05, gpu_time=0.03)
```

### 3. Generate Reports

```python
# Generate and print report
report = detector.generate_report(step=current_step)
print(report.to_text())

# Save report to file
detector.save_report(report, "straggler_report.json")

# Identify stragglers
stragglers = report.identify_stragglers(threshold=1.5)
print(f"Detected stragglers: {stragglers}")
```

## Advanced Usage

### Communication Monitoring

```python
from flagscale.runner.straggler import CommProfiler

# Initialize communication profiler
comm_profiler = CommProfiler(backend="nccl", enabled=True)

# Wrap all-reduce operation
all_reduce_func = comm_profiler.wrap_operation("all_reduce", original_all_reduce_func)

# Record custom operation
start_time = time.perf_counter()
# ... do work ...
end_time = time.perf_counter()
comm_profiler.record_custom_operation(
    op_type="broadcast",
    op_name="model_sync",
    start_time=start_time,
    end_time=end_time,
    data_size=1048576,
)

# Get statistics
stats = comm_profiler.get_stats()
stragglers = comm_profiler.get_stragglers(threshold=2.0)
```

### Network Health Check

```python
from flagscale.runner.straggler import NetworkHealthChecker

# Initialize health checker
checker = NetworkHealthChecker(rank=0, world_size=4)

# Check node connectivity
node_ips = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
health_results = checker.comprehensive_health_check(node_ips)

# Identify unhealthy nodes
unhealthy = checker.identify_unhealthy_nodes(
    health_results,
    max_latency_ms=100.0,
    min_bandwidth_mbps=10.0,
)

print(f"Unhealthy nodes: {unhealthy}")

# Save health report
checker.save_health_report(health_results, "network_health.json")
```

### Elastic Training Health Monitoring

```python
from flagscale.runner.straggler import ElasticTrainingHealthChecker

# Initialize elastic training checker
elastic_checker = ElasticTrainingHealthChecker(rank=0, world_size=4)

# Monitor health over time
health_history = elastic_checker.monitor_elastic_health(
    node_ips=node_ips,
    check_interval=30.0,
    num_checks=10,
)

# Detect unstable nodes
unstable_nodes = elastic_checker.detect_unstable_nodes(
    health_history,
    instability_threshold=0.3,
)

print(f"Unstable nodes: {unstable_nodes}")
```

### Integration with Training Loop

```python
for step in range(total_steps):
    # Increment step counter
    detector.increment_step()

    # Profile dataloader
    with OptionalSectionContext(detector, "dataloader", enabled=detector.should_profile()):
        batch = dataloader.next_batch()

    # Profile forward pass
    with OptionalSectionContext(detector, "forward", enabled=detector.should_profile()):
        output = model(batch)

    # Profile backward pass
    with OptionalSectionContext(detector, "backward", enabled=detector.should_profile()):
        loss.backward()

    # Profile optimizer
    with OptionalSectionContext(detector, "optimizer", enabled=detector.should_profile()):
        optimizer.step()

    # Generate periodic reports
    if detector.should_report():
        report = detector.generate_report(step=step)

        # Log to console
        logger.info(report.to_text())

        # Save to file
        if rank == 0:
            detector.save_report(report, f"straggler_step_{step}.json")
```

## Configuration Options

### StragglerConfig Parameters

- `enabled`: Enable/disable straggler detection
- `scores_to_compute`: "relative", "individual", or "all"
- `gather_on_rank0`: Gather all statistics on rank 0
- `profiling_interval`: Profile every N steps
- `report_interval_steps`: Generate report every N steps
- `monitor_sections`: List of sections to monitor
- `enable_comm_logging`: Enable communication logging
- `enable_gpu_profile`: Enable GPU profiling
- `straggler_threshold`: Relative slowdown factor for straggler detection
- `warmup_steps`: Ignore first N steps (warmup)

### Best Practices

1. **Use OptionalSectionContext**: Avoids overhead when profiling is disabled
2. **Set appropriate thresholds**: 1.5-2.0 is typically good for straggler detection
3. **Warmup steps**: Always skip initial steps to avoid measurement noise
4. **Sampling**: Don't profile every step (use `profiling_interval`)
5. **Reporting**: Report less frequently than profiling to reduce overhead
6. **Rank 0 aggregation**: Set `gather_on_rank0=True` for centralized reporting

## Notes

- The module is designed to work with PyTorch distributed training
- GPU profiling requires CUDA to be available
- Communication hooks provide basic monitoring - integrate with your specific backend
- Network health checks are primarily designed for training pre-checks
- All timing measurements are in seconds
- Reports are JSON-serializable for storage and analysis
