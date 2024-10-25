import asyncio
from .bench import main


def cli() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    exit(cli())
