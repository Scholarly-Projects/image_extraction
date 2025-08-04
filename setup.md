## Image Extraction Tool - Setup Guide (Mac and Windows)

This guide explains how to set up and run a Python-based tool for identifying and extracting individual photographs within digital overhead scans of archival scrapbooks and other bound resources.

---


## What You Need

- Python 3.10 or 3.11 installed
- A terminal (Command Prompt, PowerShell, or Terminal on macOS)
- GitHub project folder containing the script

---

## 1. Install Python

### Windows:

1. Visit https://www.python.org/downloads/
2. Download Python 3.11.x
3. During installation:
   - Check the box “Add Python to PATH”
   - Click “Customize installation” and ensure `pip` is included
   - Complete the installation

To verify installation, run in Command Prompt:

```sh
python --version
```

### macOS:

Install Python using Homebrew:

```sh
brew install python@3.11
```

To verify:

```sh
python3 --version
```

---

## 2. Set Up a Virtual Environment

__Mac__

```sh
python3 -m venv .venv
source .venv/bin/activate
```

__Windows__

```sh
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Required Python Libraries

```sh
pip install opencv-python numpy
```
---

## 4. Project Folder Structure

Ensure your project looks like this:

```
edge_detector/
├── script.py                 # Your image extraction script
├── A/                        # Input images (JPG, PNG, TIFF)
├── B/                        # Output: individual photos within the images
```

To create folders manually:

```sh
mkdir A B
```

Place your input images in the `A` folder.

---

## 5. Run the Script

```sh
python script.py
```

---

## 6. Deactivate 

To deactivate the virtual environment:

```sh
deactivate
```

---
