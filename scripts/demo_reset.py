import os
import sqlite3
import argparse
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("demo_reset")
settings = get_settings()

def reset_db():
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    
    if not os.path.exists(db_path):
        logger.info(f"Database file not found at {db_path}. Nothing to reset.")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count before
        cursor.execute("SELECT COUNT(*) FROM alerts")
        count_before = cursor.fetchone()[0]
        
        # Delete all
        cursor.execute("DELETE FROM alerts")
        conn.commit()
        
        logger.info(f"✅ Reset successful! Deleted {count_before} alerts from {db_path}.")
    except Exception as e:
        logger.error(f"❌ Failed to reset database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset demo state (clear alerts)")
    parser.parse_args()
    reset_db()
