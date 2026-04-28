# YT Trending Intelligence Dashboard
 
> A fully automated, cloud-deployed data science system that collects YouTube trending videos from 5 countries every 24 hours.
 
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/ashishkumar2005/yt/collect_trending.yml?label=Data%20Collection&logo=github)](https://github.com/ashishkumar2005/yt/actions)
[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-red?logo=streamlit)](https://your-streamlit-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![MySQL](https://img.shields.io/badge/Database-MySQL-orange?logo=mysql)](https://mysql.com)
 
---
 
## Live Demo
 
**Dashboard:** https://youtubetrendanalyser.streamlit.app/
 
---
 
## Screenshots
 
| Live Feed | Analysis |
|-----------|----------|
| ![Live Feed](screenshots/live_feed.png) | ![Analysis](screenshots/analysis.png) |

| Country Compare | Predict |
|-----------------|---------|
| ![Country](screenshots/country_compare.png) | ![Predict](screenshots/predict.png) |

### Database
![MySQL](screenshots/mysql.png)
 
---
 
## What This Project Does
 
Think of it like **a newspaper that writes itself.**
 
Every 24 hours, an automated pipeline:
1. **Wakes up** on GitHub Actions (no laptop needed)
2. **Calls YouTube API** → fetches top 5 trending videos from 5 countries
3. **Saves 25 videos** into a cloud MySQL database on TiDB
4. **Dashboard updates** automatically — anyone can visit the live URL
 
```
Every 24 hours (GitHub Actions)
        ↓
YouTube Data API v3
        ↓
25 videos (US · IN · GB · CA · AU)
        ↓
TiDB MySQL Database
        ↓
Live Streamlit Dashboard
```
 
---
 
## Architecture
 
```
┌─────────────────────────────────────────┐
│              GITHUB ACTIONS             │
│   Cron: every 24 hours (0 */24 * * *)    │
│   → Installs dependencies               │
│   → Runs src/data_collector.py          │
│   → Fetches 25 videos from YouTube     │
│   → Saves to MySQL                      │
└────────────────────┬────────────────────┘
                     │ inserts data
                     ↓
┌─────────────────────────────────────────┐
│                RAILWAY                  │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  MySQL DB    │  │ Streamlit App   │  │
│  │  videos      │◄─│ dashboard/      │  │
│  │  snapshots   │  │ app.py          │  │
│  └──────────────┘  └────────┬────────┘  │
└───────────────────────────  │  ─────────┘
                              ↓
                        streamlit
```
 
---
 
##  Key Features
 
- **Fully Automated** — GitHub Actions runs 24/7, no manual effort
- **5 Countries** — US, India, UK, Canada, Australia
- **25 videos per run** — 50 per country every 8 hours
- **ML Predictions** — Logistic Regression
- **5-Page Dashboard** — Live Feed, Analysis, Country Compare, Historical, Predict
- **Production Security** — All secrets managed via GitHub Secrets & Railway env vars
- **Cloud MySQL** — Persistent database with 200+ rows and growing
 
---
 
## Project Structure
 
```
yt/
├── .github/
│   └── workflows/
│       └── collect_trending.yml   # GitHub Actions scheduler
├── dashboard/
│   └── app.py                     # Streamlit dashboard (5 pages)
├── src/
│   ├── data_collector.py          # YouTube API fetcher
│   ├── database.py                # MySQL connection & queries
│   ├── data_cleaner.py            # Feature engineering (43 features)
│   └── scheduler.py               # Local scheduler (optional)
├── models/
│   ├── trend_classifier.py        # Trains Random Forest, Ridge, K-Means
│   └── saved/                     # Saved .pkl model files
├── data/
│   └── raw/                       # JSON backups of API responses
├── config.py                      # Centralised configuration
├── requirements.txt               # Python dependencies
```
 
---
 
## Machine Learning

This project uses **Logistic Regression** to predict whether a YouTube video is likely to trend.

### Model Overview
- Each feature (views, likes, comments, etc.) is assigned a weight  
- The model calculates a weighted sum of all features  
- If the score crosses a threshold, the video is predicted to trend  

### How It Works

- Uses **11 input features** from video metadata  
- Outputs a **probability (0–100%)** of a video trending  
- Classifies videos into:
  - **High Trend Potential**
  - **Low Trend Potential**

### Real-Time Training

Unlike static models, this system:

- Trains **dynamically on live database data**
- Retrains **every time the Predict page is opened**
- Always reflects the **latest trending patterns**

### Why Logistic Regression?

- Fast and efficient for real-time predictions  
- Interpretable (easy to understand feature impact)  
- Works well with structured tabular data  

### Features Used

- View Count  
- Like Count  
- Comment Count  
- Duration  
- Publish Time  
- Title Features (TF-IDF)  
- Engagement Metrics  
 
**40+ engineered features** including:
- `views_per_hour` — viral velocity
- `like_view_ratio` — audience quality signal
- `hours_to_trend` — upload-to-trending speed
- `title_sentiment` — TextBlob NLP polarity score
- `is_short` — YouTube Shorts detection
- `tag_count`, `title_length`, `publish_hour`, and more
 
---
 
## Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Data Collection | YouTube Data API v3 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression), TextBlob |
| Database | MySQL (TiDB) |
| Dashboard | Streamlit, Plotly |
| Scheduler | GitHub Actions (Scheduled Jobs) |
| Hosting | Streamlit |
| Secret Management | GitHub Secrets + Railway Env Vars |
 
---
 
## Setup & Installation
 
### Prerequisites
- Python 3.11+
- YouTube Data API v3 key ([Get one here](https://console.cloud.google.com))
- MySQL database (TiDB)
 
### 1. Clone the repository
```bash
git clone https://github.com/ashishkumar2005/yt.git
cd yt
```
 
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Configure environment variables
Create a `.env` file in the project root:
```env
YOUTUBE_API_KEY=your_api_key_here
MYSQLHOST=localhost
MYSQLPORT=3306
MYSQLUSER=root
MYSQLPASSWORD=your_password
MYSQLDATABASE=railway
```
 
### 4. Run the data collector once
```bash
python src/data_collector.py
```
 
### 5. Train the ML models
```bash
python models/trend_classifier.py
```
 
### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
 
---
 
## Automated Collection (GitHub Actions)
 
The `.github/workflows/collect_trending.yml` workflow runs automatically:
 
```yaml
on:
  schedule:
    - cron: '0 */24 * * *'   # Every 24 hours
  workflow_dispatch:          # Manual trigger anytime
```
 
Add these secrets to your GitHub repo (**Settings → Secrets → Actions**):
 
| Secret | Description |
|--------|-------------|
| `YOUTUBE_API_KEY` | Your YouTube Data API v3 key |
| `MYSQLHOST` | Railway MySQL public host |
| `MYSQLPORT` | Railway MySQL port |
| `MYSQLUSER` | Database username |
| `MYSQLPASSWORD` | Database password |
| `MYSQLDATABASE` | Database name |
 
---
 
## Key Stats
 
- ✅ **25 videos** collected per run
- ✅ **2,000+ rows** already in database and growing
- ✅ **Every 24 hours** automatically
- ✅ **5 countries** tracked simultaneously
- ✅ **43 features** engineered per video
- ✅ **~40 seconds** per full collection run
- ✅ **0 manual effort** after deployment
 
---
 
## Security
 
- `.env` file is listed in `.gitignore` — never uploaded to GitHub
- All production secrets stored in GitHub Encrypted Secrets
- Railway environment variables used for live deployment
- Database uses public proxy host for external connections
 
---
 
## Author
 
**Ashish Kumar**
- LinkedIn: [@ashishkumar2005](https://www.linkedin.com/in/ashishkumar2005/)
