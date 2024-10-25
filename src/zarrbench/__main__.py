import asyncio
from .bench import main


def cli() -> None:
    exit(asyncio.run(main()))


if __name__ == "__main__":
    exit(cli())
