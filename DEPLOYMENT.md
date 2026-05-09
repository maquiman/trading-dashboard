# Deployment Guide

This project is ready to deploy to Streamlit Community Cloud using:

- Main app file: `app.py`
- Python dependencies: `requirements.txt`

## 1. Create a GitHub Repository

1. Sign in to GitHub.
2. Click **New repository**.
3. Choose a repository name such as `trading-dashboard`.
4. Leave it as a public or private repository based on your preference.
5. Create the repository.

## 2. Push This Project to GitHub

From the project folder, run:

```powershell
git init
git add .
git commit -m "Prepare Streamlit dashboard for deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If the repository already exists locally, just add/verify the remote and push:

```powershell
git remote -v
git add .
git commit -m "Update deployment files"
git push
```

## 3. Connect the Repository to Streamlit Community Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io).
2. Sign in with GitHub.
3. Click **New app**.
4. Select the GitHub repository you pushed.
5. Select the branch to deploy, usually `main`.

## 4. Set the Main File Path

Use this as the entry point:

```text
app.py
```

If you later move the app into a subfolder, update the file path in Streamlit Community Cloud to match that location.

## 5. Configure Secrets if Needed

The current dashboard does not require API keys or Streamlit secrets to run.

If you add secrets later:

1. In Streamlit Community Cloud, open the app settings.
2. Open **Secrets**.
3. Add key/value pairs there instead of committing them to the repository.

Do not commit:

- `.streamlit/secrets.toml`
- `.env`
- API keys, tokens, or passwords

## 6. Recommended Repository Files

For deployment, keep these files in the repository:

- `app.py`
- `backtest.py`
- `levels.py`
- `signals.py`
- `vcp.py`
- `wills_signal.py`
- `help_text.py`
- `requirements.txt`
- `README.md`
- `DEPLOYMENT.md`

Optional local-only files such as logs, caches, and virtual environments are excluded by `.gitignore`.

## 7. Troubleshooting

If Streamlit Community Cloud says it cannot deploy:

1. Confirm the repository is connected to GitHub.
2. Confirm `requirements.txt` exists in the repository root.
3. Confirm the main file path is `app.py`.
4. Confirm there are no local-only paths or missing imports.
5. Check the Streamlit build logs for package install or import errors.

## 8. Notes for This Project

- The app uses a local relative cache directory (`.yfinance-cache`) at runtime. That cache is ignored by Git and does not need to be uploaded.
- The app is designed to run from the repository root.
- No deployment-specific trading logic changes were made.
