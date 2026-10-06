# GitHub setup

**Status: done automatically on 2026-10-05.** The owner authorised the GitHub CLI, so Claude
created the repo and enabled Pages. These steps are kept as a record and for re-creating the
setup by hand.

## What was done

1. Create the public repo from the local folder and push:
   ```powershell
   gh repo create ais-sentinel --public --source=. --remote=origin --push `
     --description "Maritime domain awareness from public AIS data: Kalman/IMM tracking, trajectory prediction with uncertainty, and anomaly detection, with a static web front end."
   ```
2. Enable GitHub Pages with **GitHub Actions** as the source:
   ```powershell
   gh api -X POST repos/jrhughes003/ais-sentinel/pages -f build_type=workflow
   ```
   Manual equivalent: on the repo page, go to **Settings → Pages → Build and deployment →
   Source**, then choose **GitHub Actions**.
3. Trigger the deploy workflow (it also runs on every push to `main` that touches `site/`):
   ```powershell
   gh workflow run pages.yml
   gh run watch
   ```
4. The site is live at https://jrhughes003.github.io/ais-sentinel/

## Doing it by hand (no gh)

1. On github.com, go to **New repository**. Use the name `ais-sentinel`, make it
   **Public**, and leave the README, .gitignore and licence options unticked.
2. Run `git remote add origin https://github.com/jrhughes003/ais-sentinel.git`, then
   `git push -u origin main`.
3. Complete step 2 above using the settings page.
