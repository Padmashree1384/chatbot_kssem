import os
import requests
from bs4 import BeautifulSoup

def fetch_and_save(url, filename):
    """Fetch content from a URL, extract structured text with headings, and save it to a file."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
            if tag.name in ['h1', 'h2', 'h3']:
                content.append(f"\n{tag.text.strip().upper()}\n" + "-" * len(tag.text))
            else:
                content.append(tag.text.strip())
        
        text_data = '\n'.join(content)
        
        # Ensure dataset folder exists
        os.makedirs("dataset", exist_ok=True)
        
        # Save to file
        with open(f"dataset/{filename}.txt", "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"Saved: dataset/{filename}.txt")
    else:
        print(f"Failed to fetch {url}: Status code {response.status_code}")

# List of websites to scrape
websites = {
    "kssem": "https://www.kssem.edu.in",
    "ks_polytechnic": "https://www.kspolytechnic.edu.in",
    "ksa": "https://www.kssa.edu.in"
}

# Scrape and save each website
for name, url in websites.items():
    fetch_and_save(url, name)
