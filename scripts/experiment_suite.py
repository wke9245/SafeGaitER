import copy
from opengait.utils import resolve, write_json
from scripts.train_safegait import train
from scripts.evaluate_recognition import evaluate
from scripts.evaluate_privacy import evaluate_privacy
from scripts.reproduce_tables import aggregate


def build_plan(base, suite="main"):
    experiments = []
    if suite in ("main", "all"):
        for method in (
            "raw",
            "clipping",
            "noise",
            "masking",
            "coarsening",
            "generic",
            "deepleak",
            "no_auditor",
            "safe",
        ):
            cfg = copy.deepcopy(base)
            cfg["training"]["strategy"] = method
            experiments.append((method, cfg))
    if suite in ("factors", "all"):
        for name, exponents, aggregation in [
            ("without_reliability", [0, 1, 1, 1], "product"),
            ("without_temporal", [1, 0, 1, 1], "product"),
            ("without_contribution", [1, 1, 0, 1], "product"),
            ("without_stability", [1, 1, 1, 0], "product"),
            ("additive", [1, 1, 1, 1], "sum"),
            ("maximum", [1, 1, 1, 1], "max"),
        ]:
            cfg = copy.deepcopy(base)
            cfg["risk"].update(exponents=exponents, aggregation=aggregation)
            experiments.append((name, cfg))
        cfg = copy.deepcopy(base)
        cfg["risk"]["use_confidence"] = False
        experiments.append(("without_confidence", cfg))
    if suite in ("thresholds", "all"):
        for key, values in {
            "tau1": [0.25, 0.30, 0.35, 0.40, 0.45],
            "tau2": [0.55, 0.60, 0.65, 0.70, 0.75],
            "tau_mask": [0.60, 0.65, 0.70, 0.75, 0.80],
            "sigma_max": [0.00, 0.05, 0.10, 0.20, 0.30],
        }.items():
            for value in values:
                cfg = copy.deepcopy(base)
                cfg["release"][key] = value
                experiments.append((f"{key}_{value:.2f}", cfg))
    if suite in ("backbones", "all"):
        for backbone in ("hierarchical", "gaitpart"):
            cfg = copy.deepcopy(base)
            cfg["model"]["backbone"] = backbone
            experiments.append((backbone, cfg))
    if suite in ("transfer", "all"):
        from opengait.utils import load_config

        for name in ("ou_in_domain", "ou_frozen_casia"):
            cfg = load_config("configs/safegait_er/ou_mvlp.yaml")
            if name == "ou_frozen_casia":
                cfg["release"] = copy.deepcopy(base["release"])
            cfg["policy_source"] = "casia_frozen" if name == "ou_frozen_casia" else "ou_in_domain"
            experiments.append((name, cfg))
    plans = []
    for name, cfg in experiments:
        for seed in base["seeds"]:
            config = copy.deepcopy(cfg)
            config["seed"] = seed
            config["output"] = str(resolve(base["output"]) / name / f"seed_{seed}")
            plans.append({"name": name, "seed": seed, "config": config})
    return plans


def run_suite(base, suite="main", plan_only=False):
    plans = build_plan(base, suite)
    write_json(resolve(base["output"]) / "experiment_plan.json", plans)
    if plan_only:
        print(
            "Experiment plan written: " + str(resolve(base["output"]) / "experiment_plan.json"),
            flush=True,
        )
        return plans
    shared_ou_checkpoints = {}
    reid_checkpoints = {}
    for plan in plans:
        cfg = plan["config"]
        print(f"Experiment {plan['name']} seed {plan['seed']}", flush=True)
        if plan["name"] == "ou_frozen_casia" and plan["seed"] in shared_ou_checkpoints:
            checkpoint = shared_ou_checkpoints[plan["seed"]]
        else:
            checkpoint = train(cfg)
            if plan["name"] == "ou_in_domain":
                shared_ou_checkpoints[plan["seed"]] = checkpoint
        archive = evaluate(
            cfg,
            checkpoint,
            queries=max(cfg["attacks"]["repeated_budgets"]),
            strategy=cfg["training"]["strategy"],
        )
        protocol = "matched" if cfg["dataset"] == "CASIA-B" else "stress"
        evaluate_privacy(cfg, archive, protocol=protocol, repeated=True)
        from scripts.evaluate_inversion import evaluate_inversion
        from scripts.evaluate_audit import evaluate_audit
        from scripts.profile_inference import profile

        key = (cfg["dataset"], cfg["model"]["backbone"])
        if key not in reid_checkpoints:
            reid_cfg = copy.deepcopy(cfg)
            reid_cfg["seed"] = 777
            reid_cfg["training"]["strategy"] = "raw"
            reid_cfg["output"] = str(
                resolve(base["output"]) / "reid_evaluators" / (key[0] + "_" + key[1]) / "seed_777"
            )
            reid_checkpoints[key] = train(reid_cfg)
        evaluate_inversion(
            cfg,
            archive,
            raw=cfg["training"]["strategy"] == "raw",
            reid_checkpoint=reid_checkpoints[key],
        )
        if cfg["dataset"] == "CASIA-B":
            evaluate_audit(cfg, checkpoint)
        profile(cfg, checkpoint)
    return aggregate(resolve(base["output"]), resolve(base["output"]) / "tables")
