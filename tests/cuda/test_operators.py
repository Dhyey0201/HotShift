import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from hotshift.engine.cuda import build_cuda_extension

    build_cuda_extension()


def _reference(rows, misses, source, inverse, offsets):
    unique = torch.stack([rows[s] if s >= 0 else misses[-s - 1] for s in source.tolist()]) if source.numel() else rows[:0]
    output = torch.zeros((offsets.numel() - 1, rows.shape[1]), dtype=torch.float32, device=rows.device)
    for bag in range(offsets.numel() - 1):
        start, end = int(offsets[bag]), int(offsets[bag + 1])
        if end > start:
            output[bag] = unique[inverse[start:end]].mean(dim=0)
    return output


@pytest.mark.parametrize("dimensions", [1, 31, 33, 127, 128, 129, 257])
def test_fused_pool_handles_awkward_dimensions(dimensions):
    device = "cuda"
    rows = torch.randn((4, dimensions), dtype=torch.float32, device=device)
    misses = torch.randn((1, dimensions), dtype=torch.float32, device=device)
    source = torch.tensor([0, 1, -1], dtype=torch.int32, device=device)
    inverse = torch.tensor([0, 1, 0, 2], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 3, 4], dtype=torch.int64, device=device)
    actual = torch.ops.hotshift.pool_mean(rows, misses, source, inverse, offsets)
    expected = torch.stack(((rows[0] + rows[1] + rows[0]) / 3, torch.zeros(dimensions, device=device), misses[0]))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dimensions", [31, 33, 127, 129, 257])
def test_gather_matches_descriptors_independently(dimensions):
    rows = torch.randn((5, dimensions), device="cuda")
    misses = torch.randn((3, dimensions), device="cuda")
    source = torch.tensor([4, -1, 0, -3, -2], dtype=torch.int32, device="cuda")
    gathered = torch.ops.hotshift.gather_rows(rows, misses, source)
    expected = torch.stack((rows[4], misses[0], rows[0], misses[2], misses[1]))
    torch.testing.assert_close(gathered, expected, atol=0, rtol=0)
    empty = torch.ops.hotshift.gather_rows(rows, misses, source[:0])
    assert empty.shape == (0, dimensions)


def test_pool_edge_cases_all_hit_all_miss_long_bags_and_repeats():
    generator = torch.Generator(device="cpu").manual_seed(4)
    dimensions = 65
    rows = torch.randn((64, dimensions), generator=generator).cuda()
    misses = torch.randn((64, dimensions), generator=generator).cuda()
    cases = []
    all_hit = torch.arange(16, dtype=torch.int32)
    all_miss = -(torch.arange(16, dtype=torch.int32) + 1)
    for source in (all_hit, all_miss, torch.where(torch.arange(16) % 2 == 0, all_hit, all_miss)):
        inverse = torch.randint(0, 16, (256 * 3,), generator=generator)
        offsets = torch.tensor([0, 1, 1, 256, 768], dtype=torch.int64)
        cases.append((source, inverse, offsets))
    for source, inverse, offsets in cases:
        args = (rows, misses, source.cuda(), inverse.cuda(), offsets.cuda())
        first = torch.ops.hotshift.pool_mean(*args)
        torch.testing.assert_close(first, _reference(*args), atol=1e-5, rtol=1e-4)
        for _ in range(3):
            torch.testing.assert_close(torch.ops.hotshift.pool_mean(*args), first, atol=0, rtol=0)


def test_empty_batch_and_zero_rows():
    rows = torch.zeros((2, 8), device="cuda")
    misses = torch.zeros((0, 8), device="cuda")
    empty = torch.ops.hotshift.pool_mean(
        rows, misses, torch.zeros(0, dtype=torch.int32, device="cuda"),
        torch.zeros(0, dtype=torch.int64, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"),
    )
    assert empty.shape == (0, 8)
    zeros = torch.ops.hotshift.pool_mean(
        rows, misses, torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 0], dtype=torch.int64, device="cuda"), torch.tensor([0, 2], dtype=torch.int64, device="cuda"),
    )
    torch.testing.assert_close(zeros, torch.zeros((1, 8), device="cuda"))


def test_probe_bounds_and_cache_miss():
    mapping = torch.tensor([-1, 2, 0], dtype=torch.int32, device="cuda")
    ids = torch.tensor([0, 1, 2, 9, -1], dtype=torch.int64, device="cuda")
    result = torch.ops.hotshift.probe_slots(ids, mapping)
    assert result.cpu().tolist() == [-1, 2, 0, -1, -1]
    assert torch.ops.hotshift.probe_slots(ids[:0], mapping).numel() == 0


def test_mapping_update_and_row_scatter():
    mapping = torch.full((10,), -1, dtype=torch.int32, device="cuda")
    torch.ops.hotshift.update_mapping_(
        mapping,
        torch.tensor([-1, -1], dtype=torch.int64, device="cuda"),
        torch.tensor([3, 7], dtype=torch.int64, device="cuda"),
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
    )
    assert mapping.cpu().tolist() == [-1, -1, -1, 0, -1, -1, -1, 1, -1, -1]
    torch.ops.hotshift.update_mapping_(
        mapping,
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        torch.tensor([5], dtype=torch.int64, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    assert mapping[3].item() == -1 and mapping[5].item() == 0 and mapping[7].item() == 1
    cache = torch.zeros((4, 129), device="cuda")
    source = torch.randn((6, 129), device="cuda")
    torch.ops.hotshift.scatter_rows_(
        cache, source,
        torch.tensor([5, 0, 3], dtype=torch.int64, device="cuda"),
        torch.tensor([2, 0, 3], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(cache[2], source[5], atol=0, rtol=0)
    torch.testing.assert_close(cache[0], source[0], atol=0, rtol=0)
    torch.testing.assert_close(cache[3], source[3], atol=0, rtol=0)
    assert torch.count_nonzero(cache[1]) == 0
    torch.ops.hotshift.scatter_rows_(cache, source, torch.zeros(0, dtype=torch.int64, device="cuda"), torch.zeros(0, dtype=torch.int32, device="cuda"))


def _pool_args():
    rows = torch.randn((4, 8), device="cuda")
    return [
        rows,
        torch.randn((1, 8), device="cuda"),
        torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        torch.tensor([0, 2], dtype=torch.int64, device="cuda"),
    ]


@pytest.mark.parametrize(
    ("position", "replacement", "message"),
    [
        (0, lambda t: t.cpu(), "CUDA"),
        (2, lambda t: t.to(torch.int64), "int32"),
        (3, lambda t: t.to(torch.int32), "int64"),
        (0, lambda t: t.to(torch.float64), "float32"),
        (0, lambda t: torch.randn((8, 4), device="cuda").t(), "contiguous"),
        (1, lambda t: torch.randn((1, 9), device="cuda"), "dimensions"),
        (4, lambda t: t.reshape(1, 2), "dimensions"),
        (0, lambda t: t.clone().requires_grad_(True), "inference-only"),
    ],
)
def test_pool_rejects_invalid_inputs_with_actionable_errors(position, replacement, message):
    args = _pool_args()
    args[position] = replacement(args[position])
    with pytest.raises(RuntimeError, match=message):
        torch.ops.hotshift.pool_mean(*args)
    torch.cuda.synchronize()  # no device fault was triggered


def test_other_operators_reject_cpu_and_wrong_dtype_inputs():
    mapping = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        torch.ops.hotshift.probe_slots(torch.tensor([0]), mapping)
    with pytest.raises(RuntimeError, match="int64"):
        torch.ops.hotshift.probe_slots(torch.tensor([0], dtype=torch.int32, device="cuda"), mapping)
    with pytest.raises(RuntimeError, match="lengths differ"):
        torch.ops.hotshift.update_mapping_(
            mapping, torch.tensor([1], device="cuda"), torch.tensor([1, 2], device="cuda"),
            torch.tensor([0], dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(RuntimeError, match="lengths differ"):
        torch.ops.hotshift.scatter_rows_(
            torch.zeros((2, 4), device="cuda"), torch.zeros((2, 4), device="cuda"),
            torch.tensor([0, 1], device="cuda"), torch.tensor([0], dtype=torch.int32, device="cuda"),
        )
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_cross_device_inputs_are_rejected():
    args = _pool_args()
    args[1] = args[1].to("cuda:1")
    with pytest.raises(RuntimeError, match="share a CUDA device"):
        torch.ops.hotshift.pool_mean(*args)
