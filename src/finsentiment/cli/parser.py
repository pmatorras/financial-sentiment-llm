"""Argument parser setup."""
import argparse
from finsentiment.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_REGISTRY
)
def add_common_args(parser):
    """Add arguments common to both training and evaluation."""
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument("-d", "--debug", action="store_true", help="Verbose debug logging")
    parser.add_argument('--model-type', choices=['single', 'multi'], default='multi')
    parser.add_argument('--model', choices=list(MODEL_REGISTRY.keys()), default='finbert', help='Select model architecture')
def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='finsentiment',
        description='Financial sentiment analysis with LLMs'
    )
    
    # Global arguments

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train a model')
    
    train_parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    train_parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    add_common_args(train_parser)

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--checkpoint', default=None, help='Model checkpoint')
    eval_parser.add_argument('--no-log', action='store_false', dest='write_log', help="Disable log writing")
    add_common_args(eval_parser)
    return parser
