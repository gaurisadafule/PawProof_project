import urllib.request
import os
from PIL import Image

SAMPLE_URLS = {
    "dog_a1.jpg": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?auto=format&fit=crop&w=800&q=80",
    "dog_a2.jpg": "https://images.unsplash.com/photo-1517849845537-4d257902454a?auto=format&fit=crop&w=800&q=80",
    "dog_b.jpg": "https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=800&q=80"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

print("Fetching sample canine biometric test images...")

for filename, url in SAMPLE_URLS.items():
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(filename, "wb") as f:
                f.write(response.read())
        
        with Image.open(filename) as img:
            print(f"[OK] Downloaded {filename} [{img.size[0]}x{img.size[1]} px]")
    except Exception as e:
        print(f"[FAIL] Failed to download {filename}: {e}")

print("\nReady! All sample images are saved in your project directory.")