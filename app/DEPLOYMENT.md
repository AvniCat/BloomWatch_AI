# Deployment guide — for the NANDA Town submission

Follow these steps in order. Estimated total: **25 minutes.**

## Step 1 — Push the code to your GitHub (5 min)

Open Terminal, cd into the bloomwatch-app folder, then:

```bash
cd "/Users/avnisingh/ai essentials/bloomwatch-app"

# Initialize git repo
git init
git add .
git status                    # verify .env is NOT listed
git commit -m "Initial BloomWatch AI submission"
git branch -M main
```

Now create the repo on GitHub:

1. Go to https://github.com/new
2. Repository name: **`bloomwatch-ai`**
3. Owner: **AvniCat**
4. Public
5. **Do NOT** check "Add a README" or ".gitignore" (we already have those)
6. Click "Create repository"

GitHub will show you a URL like `https://github.com/AvniCat/bloomwatch-ai.git`. Back in Terminal:

```bash
git remote add origin https://github.com/AvniCat/bloomwatch-ai.git
git push -u origin main
```

If GitHub asks for authentication, use a personal access token (Settings → Developer settings → Personal access tokens → generate → tick "repo" scope).

## Step 2 — Deploy to Render (10 min)

1. Go to https://render.com/register — sign up with GitHub (fastest — auto-connects your repos)
2. Once logged in, click **"New +" → "Web Service"**
3. Click the **`bloomwatch-ai`** repo when it appears
4. Render will auto-detect `render.yaml` and pre-fill everything. Verify:
   - Name: `bloomwatch-ai`
   - Runtime: Python
   - Plan: **Free**
5. Scroll to **Environment** section. Add one variable:
   - Key: `GEMINI_API_KEY`
   - Value: `YOUR_GEMINI_API_KEY_HERE`
6. Click **"Deploy Web Service"**
7. Watch the build log — takes ~5 min. When it says "Your service is live at https://bloomwatch-ai.onrender.com" you're done.

Copy that URL — it's what you need for the form.

## Step 3 — Verify it works (2 min)

In Terminal or browser:

```bash
# Health check
curl https://bloomwatch-ai.onrender.com/health

# Live forecast
curl https://bloomwatch-ai.onrender.com/forecast?region=Kerala

# Chat
curl -X POST https://bloomwatch-ai.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Should I harvest my mussels this week?","region":"Kerala"}'
```

First request after deploy has cold-start (~30s). Subsequent ones are ~5s.

## Step 4 — Fill the NANDA submission form (5 min)

Copy-paste these into the form fields:

| Field | Value |
|---|---|
| **Skill name** | `BloomWatch AI` |
| **Your name or team** | `Avni Singh` |
| **Email** | `neeraj.invincible@gmail.com` |
| **GitHub username** | `AvniCat` |
| **One line: what does it do?** | `Early-warning API for harmful algal blooms along Kerala + Karnataka coasts — takes a shellfish farmer's question, returns a grounded answer citing a live satellite/rainfall forecast + CMFRI historical evidence.` |
| **Submission format** | Select **"Hosted link"** (or **"GitHub repo"** if you prefer) |
| **Hosted .md link** | `https://raw.githubusercontent.com/AvniCat/bloomwatch-ai/main/SKILL.md` |
| **Your endpoints** | Paste these four lines: |

```
https://bloomwatch-ai.onrender.com/health
https://bloomwatch-ai.onrender.com/providers
https://bloomwatch-ai.onrender.com/forecast
https://bloomwatch-ai.onrender.com/chat
```

| Field | Value |
|---|---|
| **Tags** | `harmful-algal-bloom, aquaculture, india, kerala, karnataka, shellfish, satellite-remote-sensing, cmfri, imd, nasa-viirs, xgboost, rag, gemini, farmer-safety` |

Click **Submit SkillMD**. Done.

## If something breaks

- **Build fails on Render**: click the build log — usually a missing dep in requirements.txt.
- **`/chat` times out**: Render free tier sleeps after 15min idle. First call wakes it (~30s). Real users won't hit this once traffic is regular.
- **Gemini rate limit hit**: the `gemini-flash-lite-latest` model has ~15 req/min free tier. Fine for the demo.
- **`git push` says "authentication failed"**: create a personal access token instead of using your password.
