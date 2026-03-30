import argparse

from info_agent import run_information_agent


def main():
    parser = argparse.ArgumentParser(
        description="Run the entity-discovery info agent for a topic query."
    )
    parser.add_argument("query", help="Topic query to research")
    args = parser.parse_args()
    print(run_information_agent(args.query))


if __name__ == "__main__":
    main()
