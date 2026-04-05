"""CLI entry point for running the research pipeline locally."""

import argparse

from info_agent import resolve_deep_research_choice, run_information_agent
from utils.progress import CliProgressReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the entity-discovery info agent for a topic query."
    )
    parser.add_argument("query", help="Topic query to research")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--deep-research",
        dest="deep_research",
        action="store_true",
        default=None,
        help="Run a broader and more exhaustive research pass",
    )
    group.add_argument(
        "--no-deep-research",
        dest="deep_research",
        action="store_false",
        help="Run the standard research pass",
    )
    parser.add_argument(
        "--recursive-research",
        action="store_true",
        help=(
            "Run a targeted follow-up search pass that backfills missing "
            "attribute/value pairs into the original table"
        ),
    )
    parser.add_argument(
        "--lightning",
        action="store_true",
        help=(
            "Run the same static pipeline with smaller search, preview, and fetch budgets"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.lightning and args.deep_research is True:
        parser.error("--lightning cannot be combined with --deep-research")
    if args.lightning and args.recursive_research:
        parser.error("--lightning cannot be combined with --recursive-research")

    with CliProgressReporter() as progress:
        result = run_information_agent(
            args.query,
            deep_research=(
                False
                if args.lightning
                else resolve_deep_research_choice(args.deep_research)
            ),
            recursive_research=(False if args.lightning else args.recursive_research),
            lightning=args.lightning,
            progress_callback=progress.update,
            detail_callback=progress.log,
        )
        progress.complete("Completed")
    print(result)


if __name__ == "__main__":
    main()
