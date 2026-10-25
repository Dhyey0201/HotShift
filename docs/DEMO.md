# Demo walkthrough

The running UI demo uses a fictional music catalog and displays the actual execution backend. On a CPU machine, keep the CPU badge visible and explain that the cache simulator illustrates residency while the reference path computes the result.

1. Show backend, dataset shape, cache capacity, and occupancy.
2. Submit a synthetic listener history and inspect ranked fictional tracks.
3. Start stable traffic and watch request count, hit rate, queue delay, and batch timing.
4. Trigger a viral-release shift; observe the request stream and policy-driven cache occupancy.
5. Stop the scenario, reset the cache or switch policies, and compare another run.
6. Explain that the CPU chart is not GPU performance evidence. On an NVIDIA host, use only recorded profiles and benchmark artifacts for performance conclusions.

The built-in scenario lasts 20 seconds in the dashboard. CLI traces are separately seeded and saved for policy replay. No immediate hit-rate increase or latency improvement is promised.

