import subprocess
import sys
import importlib
from pathlib import Path

PROJECT = "Benchmark_TL"
GNN_ARCH = "trans_encoder"
# Loop order: Landshut last so other targets complete first (same six cities as before).
# CITIES = ["regensburg", "bayreuth", "schweinfurt", "bamberg", "wuerzburg", "landshut"]

CITIES = [ "wuerzburg","bamberg", "schweinfurt", "regensburg", "bayreuth", "landshut"]
# Run with: cd ... && nohup python -u run_benchmark_tl_crosstres.py > benchmark_tl_crosstres.log 2>&1 &

def main() -> None:
    run_models = importlib.import_module("scripts.training.run_models")

    for target in CITIES:
        sources = [city for city in CITIES if city != target]
        print(f"\n=== TARGET {target} | SOURCES {sources} ===", flush=True)

        run_models = importlib.reload(run_models)
        run_models.train_cities = sources
        run_models.val_cities = []
        # Pretraining is strictly source-only: no target-city graphs are used here.
        # Final evaluation for the target city happens after finetuning.
        run_models.test_cities = []

        sys.argv = [
            "run_models.py",
            "--project_name",
            PROJECT,
            "--gnn_arch",
            GNN_ARCH,
            "--use_inductive_variant",
            "False",
            "--unique_model_description",
            f"CrossTReS_{target}_pretrain",
            "--apply_source_city_weighting_crosstres",
            "True",
            "--num_epochs",
            "300",
            "--early_stopping_patience",
            "15",
        ]
        run_models.main()

        base = Path(run_models.base_dir) / PROJECT / f"CrossTReS_{target}_pretrain"
        ckpt_dir = base / "trained_model" / "checkpoints"
        has_ckpt = ckpt_dir.exists() and any(
            p.name.startswith("checkpoint_epoch_") and p.suffix == ".pt" for p in ckpt_dir.iterdir()
        )
        if not has_ckpt:
            raise RuntimeError(f"Missing pretrain checkpoints for {target}: {ckpt_dir}")

        for i in range(1, 6):
            seed = 41 + i
            split = (
                f"data/splits/{target}/rs_{i}/t40_v10/"
                f"{target}_rs{i}_t40_v10_seed{seed}_train40_val10_test100_distant_iou.json"
            )

            cmd = [
                "python",
                "scripts/training/finetune_models.py",
                "--project_name",
                PROJECT,
                "--gnn_arch",
                GNN_ARCH,
                "--run_name",
                f"CrossTReS_{target}_finetune_rs_{i}",
                "--pretrain_run_name",
                f"CrossTReS_{target}_pretrain",
                "--cities",
                target,
                "--start_from_scratch",
                "False",
                "--split_file",
                split,
                "--num_epochs",
                "300",
                "--early_stopping_patience",
                "15",
            ]
            subprocess.run(cmd, check=True)

    print("\nAll CrossTReS benchmark runs completed.")


if __name__ == "__main__":
    main()
