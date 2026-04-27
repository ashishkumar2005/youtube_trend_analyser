import os

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv
        try:
            load_dotenv(env_path, encoding="utf-8", override=False)
        except UnicodeDecodeError:
            load_dotenv(env_path, encoding="utf-16", override=False)
    except ImportError:
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

load_env()

# ── YouTube ──────────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# ── Collection ───────────────────────────────────────────────
# 5 videos per country × 5 countries = 25 videos per day
# This runs once per day at midnight UTC
# After 1 week  → 175 videos in database
# After 1 month → 750 videos in database
# All dashboard features work well with this amount
COUNTRIES   = ["US", "IN", "GB", "CA", "AU"]
MAX_RESULTS = 5  # change to 10 if you want more data per day

# ── MySQL / TiDB Cloud ───────────────────────────────────────
# TiDB Cloud uses port 4000 (not 3306 like local MySQL)
# TiDB Cloud requires SSL=true for security
MYSQL_HOST     = os.environ.get("MYSQLHOST",     "localhost")
MYSQL_PORT     = int(os.environ.get("MYSQLPORT",  4000))
MYSQL_USER     = os.environ.get("MYSQLUSER",     "root")
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQLDATABASE", "railway")
MYSQL_SSL      = os.environ.get("MYSQL_SSL",     "true").lower() == "true"
