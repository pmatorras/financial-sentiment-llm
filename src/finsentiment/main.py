"""Main execution logic for finsentiment commands."""

from finsentiment.cli.parser import create_parser
from finsentiment.cli import train, evaluate, analyze


def main():
    """Parse arguments and route to command handlers."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == 'train':
        train.execute(args)
    elif args.command == 'evaluate':
        evaluate.execute(args)
    elif args.command == 'analyze':
        analyze.execute(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
