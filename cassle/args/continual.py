from argparse import ArgumentParser


def continual_args(parser: ArgumentParser):
    """Adds continual learning arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """
    # base continual learning args
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--task_idx", type=int, required=True)

    SPLIT_STRATEGIES = ["class", "data", "domain"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, required=True)

    # distillation args
    parser.add_argument("--distiller", type=str, default=None)
