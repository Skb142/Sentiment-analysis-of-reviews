from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
from datetime import datetime

def scrape_ebay_feedback(feedback_url, max_pages=600):
    """Scrapes eBay feedback comments along with category, title, content, and rating."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(feedback_url)
    wait = WebDriverWait(driver, 10)

    feedback_data = []
    page = 1

    while page <= max_pages:
        print(f"📄 Scraping page {page}...")
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        try:
            feedback_elements = driver.find_elements(By.CLASS_NAME, 'fdbk-container__details__comment')
            rating_elements = driver.find_elements(By.CLASS_NAME, 'fdbk-star-rating')
            
            for i, element in enumerate(feedback_elements):
                review_content = element.text.strip()
                category = "General"  # Placeholder, adjust based on extracted data
                review_title = review_content.split('.')[0] if '.' in review_content else "No Title"
                rating = rating_elements[i].get_attribute("aria-label") if i < len(rating_elements) else "No Rating"
                
                feedback_data.append([category, review_title, review_content, rating])
        except Exception as e:
            print(f"⚠️ Error extracting feedbacks: {e}")
        
        try:
            next_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'pagination__next')]"))
            )
            driver.execute_script("arguments[0].scrollIntoView();", next_button)
            time.sleep(1)
            next_button.click()
            time.sleep(3)
            page += 1
        except Exception as e:
            print(f" No more pages available or pagination button not found. Error: {e}")
            break

    driver.quit()
    return feedback_data

def save_feedback_to_csv(feedback_data):
    """Saves extracted feedbacks to a CSV file."""
    filename = "ebay_reviews.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Review Title", "Review Content", "Rating"])
        writer.writerows(feedback_data)
    
    print(f" Feedback saved to {filename}")

#  Scraping eBay feedback
feedback_url = "https://www.ebay.com/fdbk/mweb_profile?fdbkType=FeedbackReceivedAsSeller&item_id=256687932761&username=discountcomputerdepot&filter=feedback_page%3ARECEIVED_AS_SELLER&q=256687932761&sort=RELEVANCE"
feedback_data = scrape_ebay_feedback(feedback_url, max_pages=600)

#  Save extracted feedback
save_feedback_to_csv(feedback_data)
