import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from data_collector import collect_all_countries
from database import create_tables, save_videos

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_and_save():
    """Main job — fetch trending videos and save to database."""
    logger.info("="*50)
    logger.info(f"Job started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info("="*50)

    try:
        df = collect_all_countries()

        if not df.empty:
            save_videos(df)
            logger.info(f"Job completed — {len(df)} videos saved.")
        else:
            logger.warning("Job completed but no videos were collected.")

    except Exception as e:
        logger.error(f"Job failed: {e}")


if __name__ == "__main__":
    logger.info("Initializing database...")
    create_tables()

    logger.info("Running first fetch immediately...")
    fetch_and_save()

    logger.info("Starting scheduler — will fetch every 3 hours...")
    scheduler = BlockingScheduler()
    scheduler.add_job(
        fetch_and_save,
        trigger=IntervalTrigger(hours=3),
        id="trending_job",
        name="Fetch YouTube Trending Videos",
        replace_existing=True
    )

    try:
        logger.info("Scheduler is running. Press Ctrl+C to stop.")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
        scheduler.shutdown()