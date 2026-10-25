import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from hotshift.engine.cuda import build_cuda_extension

    build_cuda_extension()


@pytest.mark.parametrize("dimensions", [31, 33, 127, 129, 257])
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


def test_probe_bounds_and_cache_miss():
    mapping = torch.tensor([-1, 2, 0], dtype=torch.int32, device="cuda")
    ids = torch.tensor([0, 1, 2, 9, -1], dtype=torch.int64, device="cuda")
    result = torch.ops.hotshift.probe_slots(ids, mapping)
    assert result.cpu().tolist() == [-1, 2, 0, -1, -1]

