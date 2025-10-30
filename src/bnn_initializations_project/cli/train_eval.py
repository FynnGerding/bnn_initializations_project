"""CLI entrypoint for training and evaluation."""

from __future__ import annotations

import argparse
import logging

from ..pipelines.run_training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the BNN across datasets.")
    parser.add_argument("--prior", required=False, default=None, help="Prior name {gaussian, laplace, student_t}.")
    parser.add_argument("--nu", type=float, default=5.0, help="Degrees of freedom for Student-t prior.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs.")
    parser.add_argument("--force", action="store_true", help="Recompute even if results exist.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args(argv)
    # if args contain prior
    if args.prior:
        print(f"Running training for prior: {args.prior}")
        result = run_training(prior=args.prior, nu=args.nu, jobs=args.jobs, force=args.force)
        logging.info("Training completed: %s", result)
    else:
        print("Running training for all priors.")
        for prior in ["isotropic_gaussian", "laplace", "student_t"]:
            print(f"Running training for prior: {prior}")
            result = run_training(prior=prior, nu=args.nu, jobs=args.jobs, force=args.force)
            logging.info("Training completed for prior %s: %s", prior, result)

if __name__ == "__main__":
    main()
