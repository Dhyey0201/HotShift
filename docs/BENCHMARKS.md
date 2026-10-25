# Benchmark protocol

No benchmark has run and no performance result is available yet. The planned comparison uses identical immutable tables, trace order, cache budget, request partitions, and initial cache state across CPU, no-cache transfer, static, LRU, decayed-frequency, and adaptive paths. A fully GPU-resident baseline is labeled as a higher-memory reference.

Formal runs will save a manifest, trace and dataset checksums, raw request/window observations, summary, configuration, environment, and memory accounting. Warmup and maintenance costs are accounted for. Host wall time and CUDA event spans are reported separately. At least three runs with declared seeds are required; unfavorable or low-benefit workloads remain in the report.

