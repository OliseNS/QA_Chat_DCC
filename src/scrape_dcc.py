"""
Simple web scraper for DCC Dialysis website.
Clean, straightforward approach to extract text content from web pages.
"""

import os
import requests
from bs4 import BeautifulSoup
import time
import re


class SimpleScraper:
    """Simple scraper that just gets text content from web pages."""
    
    def __init__(self):
        self.base_url = "https://dccdialysis.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Pages to scrape
        self.pages = [
            {"name": "home", "url": "/"},
            {"name": "about-us", "url": "/about-us/"},
            {"name": "about-us_mission-values", "url": "/about-us/mission-values/"},
            {"name": "about-us_the-dcc-story", "url": "/about-us/the-dcc-story/"},
            {"name": "about-us_leadership", "url": "/about-us/leadership/"},
            {"name": "about-us_dcc-cares", "url": "/about-us/dcc-cares/"},
            {"name": "treatments", "url": "/treatments/"},
            {"name": "treatments_starting-on-dialysis", "url": "/treatments/starting-on-dialysis/"},
            {"name": "treatments_home-hemodialysis", "url": "/treatments/home-hemodialysis/"},
            {"name": "treatments_peritoneal-dialysis", "url": "/treatments/peritoneal-dialysis/"},
            {"name": "treatments_staff-assisted-dialysis", "url": "/treatments/staff-assisted-dialysis/"},
            {"name": "treatments_in-center-hemodialysis", "url": "/treatments/in-center-hemodialysis/"},
            {"name": "find-a-dialysis-center", "url": "/find-a-dialysis-center/"},
            {"name": "find-a-dialysis-center_locations", "url": "/search-by-city/"},
            {"name": "contact-us", "url": "/contact-us/"}
        ]
        
        # Setup data directory
        self.setup_data_dir()
    
    def setup_data_dir(self):
        """Create data directories if they don't exist."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def clean_text(self, text):
        """Clean up text by removing extra whitespace."""
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_page_text(self, url):
        """Get text content from a single page."""
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Special handling for location pages - preserve structure
            if 'search-by-city' in url:
                return self.extract_location_info(soup)
            
            # Try to find main content area for other pages
            main_content = None
            for selector in ['main', '.main-content', '.content', 'article', 'body']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup
            
            # Get text
            text = main_content.get_text()
            return self.clean_text(text)
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
    
    def extract_location_info(self, soup):
        """Extract structured location information from search-by-city page."""
        locations = []
        
        # Get the full page text first
        page_text = soup.get_text()
        
        # Method 1: Look for "DIALYSIS CARE CENTER" followed by location details
        # Pattern: DIALYSIS CARE CENTER [NAME] followed by Address:, Phone:, etc.
        location_pattern = r'(DIALYSIS CARE CENTER [A-Z\s]+)\s*Address:\s*([^P]+)Phone:\s*([^F]+)(?:Fax:\s*([^S]+))?Services:\s*([^G]+)'
        matches = re.findall(location_pattern, page_text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            name = match[0].strip()
            address = match[1].strip()
            phone = match[2].strip()
            fax = match[3].strip() if match[3] else ""
            services = match[4].strip()
            
            location_info = f"""
{name}
Address: {address}
Phone: {phone}"""
            
            if fax:
                location_info += f"\nFax: {fax}"
            
            location_info += f"\nServices: {services}"
            locations.append(location_info.strip())
        
        # Method 2: If no structured matches, try to split by recognizable patterns
        if not locations:
            # Split text by "DIALYSIS CARE CENTER" entries
            parts = re.split(r'(DIALYSIS CARE CENTER [A-Z\s\-]+)', page_text, flags=re.IGNORECASE)
            
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    center_name = parts[i].strip()
                    center_details = parts[i + 1][:500].strip()  # Limit length
                    
                    # Only include if it looks like location data
                    if any(word in center_details.lower() for word in ['address', 'phone', 'services']):
                        locations.append(f"{center_name}\n{center_details}")
        
        # Join all locations with clear separators
        if locations:
            result = "DIALYSIS CENTER LOCATIONS:\n\n"
            result += "\n\n" + "="*50 + "\n\n".join(locations)
            result += "\n\n" + "="*50
            return result
        else:
            # Fallback - return cleaned text but mark it as potentially incomplete
            return f"LOCATION PAGE CONTENT (may need manual review):\n\n{self.clean_text(page_text)}"
    
    def save_page_content(self, page_name, content):
        """Save page content to text file."""
        if not content:
            print(f"No content to save for {page_name}")
            return
        
        file_path = os.path.join(self.raw_dir, f"{page_name}.txt")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Error saving {page_name}: {e}")
    
    def scrape_all(self):
        """Scrape all pages and save content."""
        print(f"Starting to scrape {len(self.pages)} pages...")
        
        for page in self.pages:
            url = self.base_url + page["url"]
            content = self.get_page_text(url)
            
            if content:
                self.save_page_content(page["name"], content)
            else:
                print(f"No content found for {page['name']}")
            
            # Be nice to the server
            time.sleep(1)
        
        print("Scraping completed!")


def main():
    """Run the scraper."""
    scraper = SimpleScraper()
    scraper.scrape_all()


if __name__ == "__main__":
    main()
