import torch

from dental_3d_vae.ml.metrics import (
    chamfer_distance,
    missing_region_error,
    observed_region_error,
    stochastic_consistency_error,
)


def test_chamfer_distance_is_symmetric_unsquared_and_zero_for_identical_clouds() -> None:
    a = torch.tensor([[[0.0, 0, 0], [1, 0, 0]]])
    b = torch.tensor([[[0.0, 0, 0]]])
    far = torch.tensor([[[3.0, 4, 0]]])

    assert chamfer_distance(a, b).tolist() == [0.5]  # a -> b: (0 + 1) / 2; b -> a: 0
    assert torch.equal(chamfer_distance(a, b), chamfer_distance(b, a))
    assert chamfer_distance(b, far).tolist() == [10.0]  # 5 + 5, not squared
    assert chamfer_distance(a, a).tolist() == [0.0]


def test_region_errors_match_a_per_tooth_loop_and_mrre_is_nan_without_removal() -> None:
    complete, reconstruction = torch.randn(3, 50, 3), torch.randn(3, 40, 3)
    removed = torch.rand(3, 50) < 0.3
    removed[2] = False

    mrre = missing_region_error(reconstruction, complete, removed)
    orpe = observed_region_error(reconstruction, complete, removed)

    for i in range(2):
        squared = torch.cdist(complete[i], reconstruction[i]).min(dim=1).values ** 2
        assert torch.allclose(mrre[i], squared[removed[i]].mean())
        assert torch.allclose(orpe[i], squared[~removed[i]].mean())
    assert mrre[2].isnan() and not orpe[2].isnan()


def test_stochastic_consistency_error_averages_pairwise_chamfer() -> None:
    a, b = torch.randn(2, 20, 3), torch.randn(2, 20, 3)  # <= 25 points: cdist is exact

    assert stochastic_consistency_error(torch.stack([a, a, a])).tolist() == [0.0, 0.0]
    assert torch.allclose(stochastic_consistency_error(torch.stack([a, b, b])), 2 / 3 * chamfer_distance(a, b))
