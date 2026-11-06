"""Use AsyncNullProtocol in asyncio services. Requires a saved 'support' agent."""

import asyncio

from nullprotocol import AsyncNullProtocol


async def main() -> None:
    async with AsyncNullProtocol() as client:
        agent = await client.load("support")
        result = await agent.summarize(
            "The parcel arrived late. One mug is cracked. The buyer wants a replacement.",
            max_length=300,
            focus="problem and requested resolution",
        )
        if result.success:
            print(result.summary, result.key_points)
        else:
            print(result.error_code, result.error)


if __name__ == "__main__":
    asyncio.run(main())
