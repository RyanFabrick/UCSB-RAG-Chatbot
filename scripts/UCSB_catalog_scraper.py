import requests # Makes HTTP requests to websites
from bs4 import BeautifulSoup # Parses HTML content, extracts specific data
import time # Delays between requests and timestamps
import os # Operating System interface
import re # Regular expressions for text pattern matching, cleaning
from urllib.parse import urljoin, urlparse # Tools for working with URLs like joining, parsing, etc
from pathlib import Path #Work with file paths
import json # Handles JSON data formats
from typing import Set, Dict, List # Type hints to lessen future confusion during RAG build hopefully...

class UCSBCatalogScraper:
    def __init__(self, base_url: str = "https://catalog.ucsb.edu", delay: float = 1.0):
        """
        Initializes the UCSB catalog scraper
        base_url: Base URL for UCSB catalog
        delay: Delay between requests
        """
        self.base_url = base_url # Main UCSB catalog URL
        self.delay = delay # Delay between requests
        self.session = requests.Session() # Session more efficeint than indiv. reqs, maintains state
        # Makes requests look like coming from real browser
        self.session.headers.update({ 
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited_urls: Set[str] = set() # Set tracking which URLs have been scraped (no duplicates)
        self.scraped_content: Dict[str, Dict] = {} # Dict storing ALL scraped data
        
    def is_valid_ucsb_url(self, url: str) -> bool:
        """
        Check if URL is a valid UCSB catalog URL to scrape
        Prevents against unwanted URL scraping
        """
        parsed = urlparse(url) # Breaks URL into scheme, netloc, path
        
        # Checks url is from catalog.ucsb.edu
        if parsed.netloc != 'catalog.ucsb.edu':
            return False
            
        # Skip certain paths
        skip_patterns = [
            '/search',
            '/login',
            '/admin',
            '.pdf',
            '.doc',
            '.zip',
            '#',
            'javascript:'
        ]
        
        # Loop iterates through skippable paths in inputted URL
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
                
        return True
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from BeautifulSoup object
        BeautifulSoup converts HTML into navigable tree structure
        """
        # Remove script and style elements (unwanted elements)
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get main content area (UCSB catalog specific)
        main_content = soup.find('main') or soup.find('div', class_='content') or soup.find('body')
        
        if main_content:
            # Extract text and clean it up, joins with spaces
            text = main_content.get_text(separator=' ', strip=True)
            # Remove extra whitespace, replaces with single spaces
            text = re.sub(r'\s+', ' ', text)
            # Removes leading or trailing whitespace too
            return text.strip()
        
        return ""
    
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """
        Extract all valid links from the current page
        Finds all links on current page for future scrapes
        """
        links = []
        
        # Finds all anchor tags w/ href attributes
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Converts URL to absolute URL
            # Ex: current_url = https://catalog.ucsb.edu/undergraduate --> result = https://catalog.ucsb.edu/href
            full_url = urljoin(current_url, href)
            
            # Validation --> only adds links that are valid UCSB URLs AND unvisited
            if self.is_valid_ucsb_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
                
        return links
    
    def scrape_page(self, url: str) -> Dict:
        """Scrape a single page and return content dictionary."""
        try:
            print(f"Scraping: {url}")
            
            # Makes web request w/ 10 sec timeout
            response = self.session.get(url, timeout=10)
            # Exceptions raised if req fails
            response.raise_for_status()

            # Parses HTML response
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract page title
            title = soup.find('title')

            if title:
                title_text = title.get_text().strip()
            else:
                title_text = "No Title"
            
            # Extract main content
            content = self.extract_text_content(soup)
            
            # Extract links for further crawling
            links = self.extract_links(soup, url)
            
            page_data = {
                'url': url,
                'title': title_text,
                'content': content,
                'links': links,
                'scraped_at': time.time()
            }
            
            return page_data
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': f"Error: {str(e)}",
                'content': "",
                'links': [],
                'scraped_at': time.time()
            }
    
    def scrape_catalog(self, max_pages: int = 100, priority_sections: List[str] = None):
        """
        Scrape the UCSB catalog starting from priority sections
        max_pages: Maximum number of pages to scrape
        priority_sections: List of URL paths to prioritize
        """
        if priority_sections is None:
            priority_sections = [
                '/undergraduate',
                '/graduate',
                '/courses',
                '/academicsandpolicies/policies',
                '/academicsandpolicies/procedures',
                '/admissions',
                '/registration'
            ]
        
        # Start with priority sections
        urls_to_visit = [urljoin(self.base_url, section) for section in priority_sections]
        
        pages_scraped = 0
        
        while urls_to_visit and pages_scraped < max_pages:
            # Gets next URL from front of queue
            current_url = urls_to_visit.pop(0)
            
            # Skips URLs already visited
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            # Scrape the page
            page_data = self.scrape_page(current_url)
            self.scraped_content[current_url] = page_data
            
            # Add new links to visit but prioritize current section
            new_links = []
            for link in page_data['links']:
                # Skip links we've already visited
                if link not in self.visited_urls:
                    new_links.append(link)
            # Add links from same section to front of queue
            # Gets parent directory of current URL -> keeps related pages together
            current_section = '/'.join(current_url.split('/')[:-1])
            # Contains links that start w/ current section path
            # Contains all other links that do NOT mathc current section
            # Sort new links by whether they're in the same section
            same_section_links = []
            other_links = []
            for link in new_links:
                # Check if link belongs to current section
                if link.startswith(current_section):
                    same_section_links.append(link)
                else:
                    other_links.append(link)
            # List creates
            urls_to_visit = same_section_links + urls_to_visit + other_links
            
            pages_scraped += 1
            
            # Dealy for rate limting to prevenet server overwhelm
            time.sleep(self.delay)
            
            print(f"Progress: {pages_scraped}/{max_pages} pages scraped")
    
    def save_to_files(self, output_dir: str = "data"):
        """Save scraped content to organized files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create subdirectories based on URL structure
        for url, data in self.scraped_content.items():
            # Create file path based on URL structure
            # Converts URL path to folder structure
            path_parts = urlparse(url).path.strip('/').split('/')
            
            if not path_parts or path_parts == ['']:
                file_path = output_path / "index.txt"
            else:
                # Create subdirectory structure
                if len(path_parts) > 1:
                    # Create nested directory structure
                    subdir_path = '/'.join(path_parts[:-1])
                    subdir = output_path / subdir_path
                else:
                    # Use root output directory
                    subdir = output_path

                # Make sure subdir exists
                subdir.mkdir(parents=True, exist_ok=True)

                # Determine filename
                if path_parts[-1]:  # If last part is not empty
                    filename = path_parts[-1]
                else:
                    filename = "index"
                
                # Gets last elemtn of list, checks is not None
                # if path_parts[-1] is not None --> last element is filename
                # else --> index as filename
                filename = path_parts[-1] if path_parts[-1] else "index"
                #joins paths with .txt extension
                file_path = subdir / f"{filename}.txt"
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {data['title']}\n")
                f.write(f"URL: {data['url']}\n")
                f.write(f"Scraped: {time.ctime(data['scraped_at'])}\n")
                f.write("-" * 50 + "\n\n")
                f.write(data['content'])
        
        # Save metadata
        with open(output_path / "scraping_metadata.json", 'w') as f:
            json.dump({
                'total_pages': len(self.scraped_content), # Count of scraped pages
                'scraped_urls': list(self.scraped_content.keys()), # Dict keys to list of URLs
                'scraping_completed': time.ctime() # Converts timestamp to readable string
            }, f, indent=2)
        
        # Fstring w/ length calculation 
        print(f"Saved {len(self.scraped_content)} pages to {output_dir}/")

def main():
    """Main function to run the scraper"""
    scraper = UCSBCatalogScraper(delay=1.0)  # 1 sec delay between requests
    
    print("Starting UCSB catalog scraping...")
    print("This will scrape key sections of the UCSB catalog.")
    print("Please be patient - this respects rate limits!\n")
    
    # Define priority sections (most important content first)
    priority_sections = [
        '/undergraduate',
        '/graduate', 
        '/courses',
        '/academicsandpolicies/policies/academicpoliciesprocedures',
        '/academicsandpolicies/policies/registration',
        '/academicsandpolicies/policies/grades',
        '/academicsandpolicies/procedures',
        '/admissions'
    ]
    
    # Start scraping
    scraper.scrape_catalog(max_pages=50, priority_sections=priority_sections)
    
    # Save results
    scraper.save_to_files("data")
    
    print(f"\nScraping complete!")
    print(f"Total pages scraped: {len(scraper.scraped_content)}")
    print(f"Content saved to: data/")

if __name__ == "__main__":
    main()