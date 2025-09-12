# Project Setup

## 1. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```
- You may need to install the PicoSDK drivers:
```bash
https://www.picotech.com/downloads
```

# Notes
## Program Running Slow
- Once when the acquisition loop was slower than usual (10s per acquisition), I restarted the kernel and re-ran it. This fixed the slower runtime.
- Unplugging the USB theoretically should wipe the memory on the PICO. Tried this and it has worked before. (Just keep it unplugged for a few seconds so any capacitors discharge in the PICO)