import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Create a database of your favorite readings and semantically search for your most-needed questions with InsightMiner."
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Database group
    db_parser = subparsers.add_parser(
        "database", help="Create or delete the local database."
    )
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    db_subparsers.add_parser(
        "create", help="Build the database from your markdown files."
    )
    db_subparsers.add_parser("clean", help="Delete the database folder.")

    # Query
    query_parser = subparsers.add_parser(
        "query", help="Ask questions to your database (or use interactive mode)."
    )
    query_parser.add_argument("question", nargs="?", help="Your question (optional).")

    args = parser.parse_args()

    # routing
    if args.command == "database":
        from database import create_database, clean_database

        if args.db_command == "create":
            create_database()
        elif args.db_command == "clean":
            clean_database()
        else:
            db_parser.print_help()

    elif args.command == "query":
        from data import query_database

        if args.question:
            query_database(args.question)
        else:
            print("Interactive mode. Type 'exit' to quit.")
            while True:
                question = input("Q> ").strip()
                if question.lower() in ["exit", "quit"]:
                    break
                if not question:
                    continue
                query_database(question)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
