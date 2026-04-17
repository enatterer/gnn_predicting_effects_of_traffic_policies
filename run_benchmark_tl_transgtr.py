import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_PROJECT = "Benchmark_TL"
DEFAULT_GNN_ARCH = "transgtr"
DEFAULT_CITIES = ["wuerzburg", "bamberg", "schweinfurt", "regensburg", "bayreuth", "landshut"]


def _pretrain_done_marker(base_dir: Path, project: str, run_name: str) -> Path:
    return base_dir / project / run_name / "trained_model" / "pretrain_completed.ok"


def _has_pretrain_artifacts(base_dir: Path, project: str, run_name: str) -> bool:
    model_path = base_dir / project / run_name / "trained_model" / "model.pth"
    marker_path = _pretrain_done_marker(base_dir, project, run_name)
    return model_path.exists() and marker_path.exists()


def _split_file_for(target: str, split_i: int, split_kind: str) -> str:
    seed = 41 + split_i
    return (
        f"data/splits/{target}/rs_{split_i}/t40_v10/"
        f"{target}_rs{split_i}_t40_v10_seed{seed}_train40_val10_test100_{split_kind}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TransGTR leave-one-city-out benchmark: "
            "pretrain on source cities, then finetune on target rs_1..rs_5."
        )
    )
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--gnn_arch", type=str, default=DEFAULT_GNN_ARCH, choices=["transgtr"])
    parser.add_argument("--cities", type=str, default=",".join(DEFAULT_CITIES))
    parser.add_argument("--only_city", type=str, default=None)
    parser.add_argument(
        "--only_split_i",
        type=int,
        default=0,
        help="0 runs all rs_1..rs_5, otherwise run one split index in 1..5.",
    )
    parser.add_argument("--split_kind", type=str, default="random", choices=["distant_iou", "random"])
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--pretrain_peak_lr", type=float, default=0.0003)
    parser.add_argument("--pretrain_initial_lr", type=float, default=0.00003)
    parser.add_argument("--finetune_peak_lr", type=float, default=0.0003)
    parser.add_argument("--finetune_initial_lr", type=float, default=0.00003)
    parser.add_argument("--use_all_features", type=str, default="False")
    parser.add_argument("--finetune_batch_size", type=int, default=32)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if len(cities) < 2:
        raise ValueError("Need at least 2 cities for leave-one-city-out benchmarking.")

    targets = cities
    if args.only_city:
        if args.only_city not in cities:
            raise ValueError(f"--only_city must be one of: {cities}")
        targets = [args.only_city]

    split_is = range(1, 6) if int(args.only_split_i) == 0 else [int(args.only_split_i)]
    for i in split_is:
        if i not in {1, 2, 3, 4, 5}:
            raise ValueError("--only_split_i must be 0 (all) or one of 1..5.")

    run_models = importlib.import_module("scripts.training.run_models")

    for target in targets:
        sources = [city for city in cities if city != target]
        print(f"\n=== TARGET {target} | SOURCES {sources} ===", flush=True)

        run_models = importlib.reload(run_models)
        run_models.train_cities = sources
        run_models.val_cities = []
        run_models.test_cities = []

        pretrain_run_name = f"TransGTR_{args.gnn_arch}_{target}_pretrain"
        # Prevent stale artifacts from previous runs from being interpreted as a fresh successful pretrain.
        pretrain_dir = Path(run_models.base_dir) / args.project / pretrain_run_name
        if pretrain_dir.exists():
            import shutil
            shutil.rmtree(pretrain_dir)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] PRETRAIN_START target={target} run={pretrain_run_name}",
            flush=True,
        )
        sys.argv = [
            "run_models.py",
            "--project_name",
            args.project,
            "--gnn_arch",
            args.gnn_arch,
            "--use_inductive_variant",
            "False",
            "--unique_model_description",
            pretrain_run_name,
            "--limit_available_graphs",
            "0",
            "--transductive_val_ratio",
            "0.2",
            "--transductive_test_ratio",
            "0.0",
            "--num_epochs",
            str(args.num_epochs),
            "--early_stopping_patience",
            str(args.early_stopping_patience),
            "--peak_lr",
            str(args.pretrain_peak_lr),
            "--initial_lr",
            str(args.pretrain_initial_lr),
            "--use_gradient_clipping",
            "True",
            "--use_all_features",
            args.use_all_features,
        ]
        run_models.main()

        base_dir = Path(run_models.base_dir)
        marker_path = _pretrain_done_marker(base_dir, args.project, pretrain_run_name)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            f"pretrain_completed_at={datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] PRETRAIN_DONE target={target} run={pretrain_run_name}",
            flush=True,
        )

        if not _has_pretrain_artifacts(base_dir, args.project, pretrain_run_name):
            model_path = base_dir / args.project / pretrain_run_name / "trained_model" / "model.pth"
            raise RuntimeError(
                f"Pretrain completion artifacts missing for {target}: model={model_path}, marker={marker_path}"
            )

        for i in split_is:
            split = _split_file_for(target=target, split_i=i, split_kind=args.split_kind)
            finetune_run_name = f"TransGTR_{args.gnn_arch}_{target}_finetune_rs_{i}"
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] FINETUNE_START target={target} split=rs_{i} run={finetune_run_name}",
                flush=True,
            )
            cmd = [
                args.python_exe,
                "scripts/training/finetune_models.py",
                "--project_name",
                args.project,
                "--gnn_arch",
                args.gnn_arch,
                "--run_name",
                finetune_run_name,
                "--unique_model_description",
                finetune_run_name,
                "--pretrain_run_name",
                pretrain_run_name,
                "--cities",
                target,
                "--start_from_scratch",
                "False",
                "--split_file",
                split,
                "--use_all_features",
                args.use_all_features,
                "--batch_size",
                str(args.finetune_batch_size),
                "--num_epochs",
                str(args.num_epochs),
                "--early_stopping_patience",
                str(args.early_stopping_patience),
                "--peak_lr",
                str(args.finetune_peak_lr),
                "--initial_lr",
                str(args.finetune_initial_lr),
            ]
            subprocess.run(cmd, check=True, env={**dict(os.environ), "PYTHONUNBUFFERED": "1"})
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] FINETUNE_DONE target={target} split=rs_{i} run={finetune_run_name}",
                flush=True,
            )

    print("\nAll TransGTR benchmark runs completed.")


if __name__ == "__main__":
    main()
