import argparse
import subprocess


DEFAULT_PROJECT = "Benchmark_TL"
DEFAULT_CITIES = ["wuerzburg", "bamberg", "schweinfurt", "regensburg", "bayreuth", "landshut"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-city-out benchmark for CityTrans (end-to-end domain-adversarial)."
    )
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--cities", type=str, default=",".join(DEFAULT_CITIES))
    parser.add_argument(
        "--only_city",
        type=str,
        default=None,
        help="If set, run only this target city (useful for debugging).",
    )
    parser.add_argument(
        "--only_split_i",
        type=int,
        default=0,
        help="If set to 1..5, run only that rs_i split (useful for debugging). 0 runs all 1..5.",
    )
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--peak_lr", type=float, default=0.0003)
    parser.add_argument("--initial_lr", type=float, default=0.00003)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_all_features", type=str, default="False")
    parser.add_argument("--split_kind", type=str, default="random", choices=["distant_iou", "random"])
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if len(cities) < 2:
        raise ValueError("Need at least 2 cities.")

    targets = cities
    if args.only_city:
        if args.only_city not in cities:
            raise ValueError(f"--only_city must be one of: {cities}")
        targets = [args.only_city]

    split_is = range(1, 6) if int(args.only_split_i) == 0 else [int(args.only_split_i)]
    for i in split_is:
        if i not in {1, 2, 3, 4, 5}:
            raise ValueError("--only_split_i must be 0 (all) or one of 1..5.")

    for target in targets:
        sources = [c for c in cities if c != target]
        print(f"\n=== TARGET {target} | SOURCES {sources} ===", flush=True)

        for i in split_is:
            seed = 41 + i
            split = (
                f"data/splits/{target}/rs_{i}/t40_v10/"
                f"{target}_rs{i}_t40_v10_seed{seed}_train40_val10_test100_{args.split_kind}.json"
            )
            run_name = f"CityTrans_{target}_rs_{i}"

            cmd = [
                "python",
                "scripts/training/train_citytrans.py",
                "--project_name",
                args.project,
                "--run_name",
                run_name,
                "--target_city",
                target,
                "--source_cities",
                ",".join(sources),
                "--split_file",
                split,
                "--num_epochs",
                str(args.num_epochs),
                "--early_stopping_patience",
                str(args.early_stopping_patience),
                "--peak_lr",
                str(args.peak_lr),
                "--initial_lr",
                str(args.initial_lr),
                "--batch_size",
                str(args.batch_size),
                "--use_all_features",
                args.use_all_features,
            ]
            subprocess.run(cmd, check=True)

    print("\nAll CityTrans benchmark runs completed.")


if __name__ == "__main__":
    main()

