import asyncio
import logging
import time
from typing import Dict, List
from backend.services.stream_reader import FFmpegStreamReader
from backend.services.redis_queue import SentinelRedisQueue

logger = logging.getLogger("sentinel.stream_ingest")


def run_ingest_worker(
    group_id: int,
    cam_ids: List[str],
    cam_urls: Dict[str, str],
    config: dict,
    redis_url: str,
):
    """
    Ingests frames for a group of cameras using FFmpegStreamReader and pushes to SentinelRedisQueue.
    """
    async def _async_loop():
        queue = SentinelRedisQueue(redis_url)
        await queue.connect()

        readers = {}
        for cid in cam_ids:
            url = cam_urls.get(cid, f"https://live.corp8.cloud/stream/{cid}")
            readers[cid] = FFmpegStreamReader(url, cid, width=1280, height=720, fps=5)
            readers[cid].start()

        logger.info(f"Stream Ingest Group {group_id} started ({len(cam_ids)} cameras)")

        target_interval = 1.0 / 5.0  # 5 FPS
        try:
            while True:
                start_time = time.time()
                for cid, reader in readers.items():
                    ret, frame = reader.read_frame()
                    if ret and frame is not None:
                        await queue.push_frame(cid, frame, time.time())

                elapsed = time.time() - start_time
                sleep_dur = max(0.01, target_interval - elapsed)
                await asyncio.sleep(sleep_dur)
        finally:
            for reader in readers.values():
                reader.stop()
            await queue.close()

    asyncio.run(_async_loop())
