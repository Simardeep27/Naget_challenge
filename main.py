import argparse

from info_agent import resolve_deep_research_choice, run_information_agent


def main():
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
    args = parser.parse_args()
    print(
        run_information_agent(
            args.query,
            deep_research=resolve_deep_research_choice(args.deep_research),
        )
    )


if __name__ == "__main__":
    main()
