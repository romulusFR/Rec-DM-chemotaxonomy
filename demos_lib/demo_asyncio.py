""" tests/demo asyncio"""

import asyncio


async def main():
    """demo"""
    await asyncio.sleep(1)
    print("hello")


if __name__ == "__main__":
    # asyncio.run(main())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
