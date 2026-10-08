"""Command-line interface for SafeGait-ER research experiments."""

import argparse
import copy
import sys


STRATEGIES = (
    "safe",
    "no_auditor",
    "raw",
    "clipping",
    "noise",
    "masking",
    "coarsening",
    "generic",
    "deepleak",
)


def parser():
    command_line = argparse.ArgumentParser(description="SafeGait-ER research experiments")
    commands = command_line.add_subparsers(dest="command")
    for command in (
        "train",
        "recognition",
        "privacy",
        "inversion",
        "audit",
        "tables",
        "sweep",
        "profile",
    ):
        options = commands.add_parser(command)
        options.add_argument(
            "--config",
            help="YAML configuration; defaults to CASIA-B or the supplied artifact configuration",
        )
        options.add_argument("--data-root")
        options.add_argument("--auxiliary-root")
        options.add_argument("--output")
        options.add_argument("--checkpoint")
        options.add_argument("--seed", type=int)
        options.add_argument("--device", choices=("auto", "cpu", "cuda"))
        if command == "train":
            options.add_argument("--resume")
            options.add_argument("--epochs", type=int)
            options.add_argument("--strategy", choices=STRATEGIES)
            options.add_argument("--backbone", choices=("hierarchical", "gaitpart"))
        if command == "recognition":
            options.add_argument("--queries", type=int, default=1)
            options.add_argument("--strategy", choices=STRATEGIES)
        if command in ("privacy", "inversion"):
            options.add_argument("--inputs")
        if command == "privacy":
            options.add_argument("--protocol", choices=("matched", "stress"))
            options.add_argument("--repeated", action="store_true")
        if command == "inversion":
            options.add_argument("--raw", action="store_true")
            options.add_argument("--reid-checkpoint")
        if command == "profile":
            options.add_argument("--repetitions", type=int, default=100)
        if command == "tables":
            options.add_argument("--results-root", default="outputs")
        if command == "sweep":
            options.add_argument("--plan-only", action="store_true")
            options.add_argument(
                "--suite",
                choices=("main", "factors", "thresholds", "backbones", "transfer", "all"),
                default="main",
            )
    return command_line


def main(argv=None):
    command_line = parser()
    args = command_line.parse_args(argv)
    if args.command is None:
        command_line.print_help()
        return
    try:
        from opengait.utils import load_config, load_checkpoint, resolve
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing dependency: {error.name}. Install the environment with "
            "python -m pip install -r requirements.txt."
        ) from error

    config = load_config(args.config or "configs/safegait_er/casia_b.yaml")
    if args.config is None:
        artifact = None
        if args.command == "train" and args.resume:
            artifact = args.resume
        elif args.command in ("privacy", "inversion"):
            artifact = args.inputs or resolve(args.output or config["output"]) / "attack_inputs.pt"
        elif args.command in ("recognition", "audit", "profile"):
            artifact = args.checkpoint or resolve(args.output or config["output"]) / "checkpoint.pt"
        if artifact and resolve(artifact).is_file():
            config = copy.deepcopy(load_checkpoint(artifact, "cpu")["config"])
    for option in ("data_root", "auxiliary_root", "output", "seed", "device"):
        value = getattr(args, option, None)
        if value is not None:
            config[option] = value

    if args.command == "train":
        from scripts.train_safegait import train

        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.strategy:
            config["training"]["strategy"] = args.strategy
        if args.backbone:
            config["model"]["backbone"] = args.backbone
        return train(config, resume=args.resume)

    checkpoint = args.checkpoint or resolve(config["output"]) / "checkpoint.pt"
    if args.command == "recognition":
        from scripts.evaluate_recognition import evaluate

        return evaluate(config, checkpoint, queries=args.queries, strategy=args.strategy)
    if args.command == "privacy":
        from scripts.evaluate_privacy import evaluate_privacy

        protocol = args.protocol or ("matched" if config["dataset"] == "CASIA-B" else "stress")
        return evaluate_privacy(
            config, inputs=args.inputs, protocol=protocol, repeated=args.repeated
        )
    if args.command == "inversion":
        from scripts.evaluate_inversion import evaluate_inversion

        return evaluate_inversion(
            config, inputs=args.inputs, raw=args.raw, reid_checkpoint=args.reid_checkpoint
        )
    if args.command == "audit":
        from scripts.evaluate_audit import evaluate_audit

        return evaluate_audit(config, checkpoint)
    if args.command == "profile":
        from scripts.profile_inference import profile

        return profile(config, checkpoint, repetitions=args.repetitions)
    if args.command == "tables":
        from scripts.reproduce_tables import aggregate

        return aggregate(resolve(args.results_root), resolve(config["output"]) / "tables")
    if args.command == "sweep":
        from scripts.experiment_suite import run_suite

        return run_suite(config, args.suite, args.plan_only)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
