import os

def load_env(filepath=".env"):
    """Load .env file — handles both UTF-8 and UTF-16 (Windows)."""
    if not os.path.exists(filepath):
        return
    for encoding in ["utf-8", "utf-16"]:
        try:
            with open(filepath, encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
            return
        except (UnicodeDecodeError, UnicodeError):
            continue

load_env()

# ── YouTube ──
API_KEY     = os.environ.get("YOUTUBE_API_KEY")
COUNTRIES   = ["US", "IN", "GB", "CA", "AU"]
MAX_RESULTS = 50

# ── MySQL — Railway native variable names (no underscore) ──
MYSQL_HOST     = os.environ.get("MYSQLHOST",     "localhost")
MYSQL_PORT     = int(os.environ.get("MYSQLPORT", 3306))
MYSQL_USER     = os.environ.get("MYSQLUSER",     "root")
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQLDATABASE", "railway")

# ── Paths ──
RAW_DATA_PATH       = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODELS_PATH         = "models/saved/"