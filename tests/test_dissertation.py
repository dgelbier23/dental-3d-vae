import numpy as np
import pandas as pd
import torch

from dental_3d_vae.experiments.dissertation import load_dissertation_run


def test_dissertation_checkpoint_loads_in_this_repos_formats(tmp_path) -> None:
    checkpoint = {  # The dissertation's keys, as stored by a complete-to-complete run.
        "config": {"encoder": "dgcnnv2"},
        "test_metadata": pd.DataFrame({"sample_id": [7, 3], "tooth_id": [11, 16]}, index=[5, 9]),
        "test_deterministic_chamfer_per_sample": np.array([0.05, 0.07]),
        "test_stochastic_chamfer_per_sample": np.array([0.06, 0.08]),
        "test_mrre_per_sample": None, "test_orpe_per_sample": None,
        "history": {"train_total": [0.3, 0.2], "train_stochastic_chamfer": [0.29, 0.19], "train_kl": [1.0, 2.0],
                    "train_feature_transform": [0.0, 0.0], "val_deterministic_chamfer": [0.1, 0.09],
                    "val_kl": [1.5, 2.5]},
        "best_epoch": 2, "epochs_trained": 2, "num_parameters": 10, "training_time_minutes": 1.5,
        "gpu_name": "A100", "verified_kl": 2.4,
    }
    torch.save(checkpoint, tmp_path / "COMPLETE_D1_test.pt")

    run = load_dissertation_run(tmp_path / "COMPLETE_D1_test.pt")

    assert run["name"] == "COMPLETE_D1_test" and run["config"]["encoder"] == "dgcnnv2"
    assert list(run["results"].columns) == ["sample_id", "tooth_id", "cd", "stochastic_cd"]  # no mrre/orpe
    assert run["results"]["cd"].tolist() == [0.05, 0.07]
    assert run["metrics"][1] == {"epoch": 2, "train_total": 0.2, "train_chamfer": 0.19, "train_kl": 2.0,
                                 "train_feature_transform": 0.0, "val_chamfer": 0.09, "val_kl": 2.5}
    assert run["summary"]["test_kl"] == 2.4
