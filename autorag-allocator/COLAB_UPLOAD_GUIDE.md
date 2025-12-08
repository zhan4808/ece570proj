# How to Upload Files to Colab

Since your GitHub repo is private, you'll need to upload files manually. Here are the easiest methods:

## Method 1: Upload Folder via Colab UI (Recommended)

1. **In Colab notebook**, click the **📁 folder icon** in the left sidebar
2. Click **"Upload to session storage"** button
3. Navigate to your local `autorag-allocator` folder
4. **Select the entire folder** and upload
5. Files will be uploaded to `/content/autorag-allocator/`

**Note**: This uploads to session storage (temporary). Files are lost when session ends, but results can be downloaded.

## Method 2: Zip and Upload

1. **On your local machine**, create a zip file:
   ```bash
   cd /Users/robertzhang/Documents/GitHub/ece570proj
   zip -r autorag-allocator.zip autorag-allocator/
   ```

2. **In Colab**, upload the zip:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select autorag-allocator.zip
   ```

3. **Extract**:
   ```python
   !unzip -q autorag-allocator.zip -d /content/
   ```

## Method 3: Use Google Drive (Persistent)

1. **Upload to Google Drive**:
   - Upload your `autorag-allocator` folder to Google Drive
   - Or sync via Google Drive desktop app

2. **Mount Drive in Colab**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Copy to working directory**:
   ```python
   !cp -r /content/drive/MyDrive/autorag-allocator /content/
   ```

## Method 4: Use Git with Personal Access Token

If you want to use git, create a personal access token:

1. **GitHub**: Settings > Developer settings > Personal access tokens > Generate new token
2. **Grant repo access**
3. **In Colab**:
   ```python
   !git clone https://YOUR_TOKEN@github.com/zhan4808/ece570proj.git
   ```

## Recommended Workflow

1. **First time**: Use Method 1 (upload folder via UI) - fastest
2. **For persistence**: Use Method 3 (Google Drive) - files survive sessions
3. **For development**: Use Method 4 (git with token) - easy updates

## After Upload

Once files are uploaded, the notebook will automatically detect them in:
- `/content/autorag-allocator/` (if uploaded directly)
- `/content/ece570proj/autorag-allocator/` (if cloned from git)

The notebook checks both locations automatically!

