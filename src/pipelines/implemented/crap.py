from random import randint
from time import sleep
import asyncio

def long_sync_task(index: int) -> str:
    processing_time = randint(1,10)
    print(f"{processing_time} started.")
    sleep(processing_time)
    print(f"{processing_time} ended.")
    return "res"*processing_time

async def async_wrapper(*args, **kwargs):
    return long_sync_task(*args, **kwargs)

async def g():
    tasks = [asyncio.create_task(async_wrapper(i)) for i in range(10)]
    results = [await t for t in tasks]
    return results


def run_all():
    asyncio.run(g())

def main():
    run_all()


if __name__=="__main__":
    main()