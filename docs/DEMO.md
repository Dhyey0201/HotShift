# Three-minute demo

Start the API (`hotshift serve --config configs/demo.yaml --backend cuda` on a GPU host, or `--backend cpu` with `configs/smoke.yaml`) and the dashboard (`npm run dev --prefix frontend`). The backend badge shows only what the API reports.

1. **Backend and budget (20 s).** Point to the badge, the device name, and the engine card: table N × D, the cache budget (an imposed fraction of the table), resident rows, and device tensor MiB. On CPU, read the amber CPU development mode banner aloud.
2. **A listener history (30 s).** Pick a profile, click *Find recommendations*, and open one track's metadata. Explain that each ID is a table row looked up through the cache, mean-pooled, then scored against 512 fictional candidates. Point out the queue, batch, and scoring times. Click again: those rows are now resident, so the window hit rate rises.
3. **Stable traffic (30 s).** Set seed 42, 20 s, 20 req/s, *Stable popularity*, and start. Watch the window hit rate settle and the churn bars shrink. Note that the policy and reset controls are locked.
4. **Viral release (45 s).** Reset the cache, choose *Viral release · abrupt shift*, and start. At the dashed *viral release* marker the hit rate drops, churn spikes, and the adaptive policy's mode may switch to `shift` (engine card). Describe whatever happens; do not promise a recovery the run did not show.
5. **Recorded comparison (30 s).** Scroll to *Policy comparison*, select a full `bench-cuda` run, and compare policies on the abrupt scenario: hit rate, throughput, batch p99, H2D bytes, recovery. Point to a row where a simpler policy wins, and note that `torch_gpu_resident` uses the full table's memory.
6. **One profiling finding (25 s).** Open `docs/PERFORMANCE.md` and state one measured result from the `ablation`, `bench-kernels`, or `profile-torch` runs, for example which stage dominates batch time. If those runs do not exist yet, say so.

Record the screen only after the GPU engine and dashboard are working end to end. Live numbers are illustrative; formal claims come from offline replay.
