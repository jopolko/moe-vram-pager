"""`obench-interp` command-line entry point.

    obench-interp doctor              env / GPU / weight readiness
    obench-interp pull [--instruct]   download gemma-2-2b + Gemma Scope SAE
    obench-interp list                phases and their status
    obench-interp run exp1 [opts]     experiment 1: multilingual concept sharing
    obench-interp run exp2 [opts]     experiment 2: planning ahead (rhyming couplets)
    obench-interp run exp3 [opts]     experiment 3: chain-of-thought faithfulness (needs --instruct)
"""
from __future__ import annotations

import argparse
import sys


def _cmd_doctor(_args: argparse.Namespace) -> int:
    from .env import run_all

    ok = True
    print("openbench-toolkit / interpretability  --  doctor\n")
    for c in run_all():
        mark = "ok  " if c.ok else "FAIL"
        print(f"  [{mark}] {c.name:22} {c.detail}")
        ok = ok and c.ok
    print("\n" + ("all green" if ok else "some checks failed (see above)"))
    return 0 if ok else 1


def _cmd_pull(args: argparse.Namespace) -> int:
    from .pull import pull

    pull(include_instruct=args.instruct)
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    from .env import _model_present, _sae_present
    from .config import INSTRUCT, ModelConfig

    rows = [
        ("1", "multilingual concept sharing", "run exp1", ModelConfig(), True),
        ("2", "planning-ahead / activation patching", "run exp2", ModelConfig(), True),
        ("3", "chain-of-thought faithfulness", "run exp3", INSTRUCT, True),
    ]
    print("phase  status        experiment")
    for n, name, cmd, cfg, ready in rows:
        w = "weights ok" if (_model_present(cfg) and _sae_present(cfg)) else "no weights"
        state = f"{'ready' if ready else 'todo':<9} {w}"
        print(f"  {n}    {state:<22} {name}  ({cmd})")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.experiment == "exp1":
        from .experiments import exp1_multilingual

        exp1_multilingual.run(args)
        return 0
    if args.experiment == "exp2":
        from .experiments import exp2_planning

        exp2_planning.run(args)
        return 0
    if args.experiment == "exp3":
        from .experiments import exp3_cot_faithfulness

        exp3_cot_faithfulness.run(args)
        return 0
    print(f"unknown experiment: {args.experiment}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="obench-interp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="env / GPU / weight readiness").set_defaults(func=_cmd_doctor)

    pp = sub.add_parser("pull", help="download weights + SAE")
    pp.add_argument("--instruct", action="store_true", help="also pull gemma-2-2b-it (phase 3)")
    pp.set_defaults(func=_cmd_pull)

    sub.add_parser("list", help="phases and status").set_defaults(func=_cmd_list)

    rp = sub.add_parser("run", help="run an experiment")
    rsub = rp.add_subparsers(dest="experiment", required=True)

    e1 = rsub.add_parser("exp1", help="multilingual concept sharing")
    from .experiments.exp1_multilingual import add_args as add_args1

    add_args1(e1)

    e2 = rsub.add_parser("exp2", help="planning ahead (rhyming couplets)")
    from .experiments.exp2_planning import add_args as add_args2

    add_args2(e2)

    e3 = rsub.add_parser("exp3", help="chain-of-thought faithfulness (needs gemma-2-2b-it)")
    from .experiments.exp3_cot_faithfulness import add_args as add_args3

    add_args3(e3)

    rp.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
