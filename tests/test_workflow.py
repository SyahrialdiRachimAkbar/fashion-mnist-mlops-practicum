import copy
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fashion_mlops.core import read_json, write_json
from fashion_mlops.workflow import assert_not_finalized, load_state


class WorkflowSafetyTests(unittest.TestCase):
    def test_test_stage_lock_blocks_training(self):
        for state in [
            {"final_evaluation": {"f1": 0.9}},
            {"final_evaluation": None, "selection_lock": {"champion_run_id": "source"}},
        ]:
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                assert_not_finalized(state)

    def test_new_data_or_config_cannot_reuse_state(self):
        with TemporaryDirectory() as folder, patch("fashion_mlops.workflow.STATE", Path(folder) / "state.json"):
            config = {"epochs": 5}
            data = {"data_md5": "example-a", "split_sha256": "split-a"}
            initial = load_state(config, data)
            self.assertEqual(initial, load_state(config, data))
            with self.assertRaises(RuntimeError):
                load_state({"epochs": 6}, data)
            with self.assertRaises(RuntimeError):
                load_state(config, {**data, "data_md5": "example-b"})

    def test_report_does_not_invent_results_when_no_runs_exist(self):
        from fashion_mlops.reporting import report
        config = {
            "dataset": {"train_size": 12000, "val_size": 3000, "split_seed": 42},
            "training": {"epochs": 5},
            "tuning": {"completed_trials": 20, "sampler_seed": 42, "lr_min": 0.0001, "lr_max": 0.003, "batch_sizes": [64, 128], "dropouts": [0.2, 0.35, 0.5]},
            "seeds": [42, 123, 2026],
        }
        with TemporaryDirectory() as folder, patch("fashion_mlops.reporting.ROOT", Path(folder)), patch("fashion_mlops.reporting.STATE", Path(folder) / "state.json"):
            destination = report(config)
            text = Path(destination).read_text(encoding="utf-8")
            self.assertIn("Belum ada training run selesai", text)
            self.assertNotIn("## Kesimpulan", text)
            self.assertFalse(read_json(Path(folder) / "reports/status.json")["experiments_complete"])

    def test_test_data_requires_explicit_final_evaluation_access(self):
        from fashion_mlops.data import load_split
        with self.assertRaisesRegex(RuntimeError, "Test set"):
            load_split("test")


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch/torchvision belum terpasang")
class LearningTests(unittest.TestCase):
    def test_seed_reproduces_training_update_and_checkpoint_predictions(self):
        import torch
        from fashion_mlops.learning import FashionCNN, set_seed

        def one_update():
            set_seed(42, 2)
            model = FashionCNN(0.3)
            images = torch.rand(4, 1, 28, 28)
            labels = torch.tensor([0, 1, 2, 3])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            return model.eval(), images

        first, images = one_update()
        second, _ = one_update()
        with torch.inference_mode():
            self.assertTrue(torch.equal(first(images), second(images)))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(first.state_dict(), path)
            restored = FashionCNN(0.3).eval()
            restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            with torch.inference_mode():
                self.assertTrue(torch.equal(first(images), restored(images)))

    def test_validation_transform_preserves_pixels_and_train_statistics(self):
        import numpy as np
        import torch
        from fashion_mlops.learning import Images

        pixels = np.full((2, 28, 28), 255, dtype=np.uint8)
        dataset = Images(pixels, np.array([3, 7]), {"mean": 0.5, "std": 0.25})
        image, label = dataset[0]
        self.assertTrue(torch.equal(image, torch.full((1, 28, 28), 2.0)))
        self.assertEqual(label.item(), 3)


if __name__ == "__main__":
    unittest.main()
