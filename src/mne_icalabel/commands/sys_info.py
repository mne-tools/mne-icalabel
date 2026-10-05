import argparse

from .. import sys_info


def run():
    """Run sys_info() command."""
    parser = argparse.ArgumentParser(
        prog=f"{__package__.split('.')[0]}-sys_info", description="sys_info"
    )
    parser.add_argument(
        "--developer",
        help="display information for optional dependencies",
        action="store_true",
    )
    args = parser.parse_args()

    sys_info(developer=args.developer)
