import requests
import json
import pandas as pd
import time
from datetime import datetime
import ollama  # For local LLM queries
import re
from bs4 import BeautifulSoup

# Create an empty DataFrame to store the final results
results_df = pd.DataFrame(columns=[
    'company_name', 'stock_name', 'filing_time', 'new_product', 'product_description'
])

def extract_section(html_content):
    # Parse the XML/HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    html_text = soup.get_text(separator=' ', strip=True)
    return html_text.replace(" ", " ").replace(" ", " ").replace(" ", " ").replace(" ", " ").replace(" ☐ ", " ")

# Function to fetch 8-K filings for a company
def get_8k_filings(cik):
    search_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(search_url, headers={'User-Agent': 'as872832@ucf.edu'})
        response.raise_for_status()
        filings_data = response.json()

        # Extract recent form 8-K filings
        if 'filings' in filings_data and 'recent' in filings_data['filings']:
            recent_filings = filings_data['filings']['recent']

            # Filter for 8-K forms
            form_indices = [i for i, form in enumerate(recent_filings.get('form', [])) if form == '8-K']

            if form_indices:
                filing_dates = [recent_filings.get('filingDate', [])[i] for i in form_indices]
                accession_numbers = [recent_filings.get('accessionNumber', [])[i] for i in form_indices]

                # Return list of filing dates and accession numbers for 8-K filings
                return list(zip(filing_dates, accession_numbers))

        return []
    except Exception as e:
        print(f"Error fetching 8-K filings for CIK {cik}: {e}")
        return []

    # Function to fetch the content of an 8-K filing


def get_filing_content(accession_number, cik):
    accession_number_clean = accession_number.replace('-', '')
    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_clean}/{accession_number}.txt"

    try:
        response = requests.get(filing_url, headers={'User-Agent': 'as872832@ucf.edu'})
        response.raise_for_status()
        content = response.text

        # Extract main body of the filing (after <TEXT> tag and before </TEXT> tag)
        match = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL)
        if match:
            return extract_section(match.group(1))
        return extract_section(content)
    except Exception as e:
        print(f"Error fetching filing content for accession {accession_number}: {e}")
        return None

    # Function to extract product information using LLM


def extract_product_info(filing_content, company_name, ticker):
    if not filing_content:
        return None

        # Prepare prompt for the LLM
    prompt = f""" 
    Analyze this SEC 8-K filing and extract information about any new product releases or announcements. 

    Company: {company_name} 
    Ticker: {ticker} 

    Task: Extract the following information in a structured format: 
    1. New Product Name: [Name of the new product] 
    2. Product Description: [Brief description of the product, less than 180 characters] 

    If no new product is mentioned in the filing, respond with "No new product found". 

    Filing content: 
    {filing_content}  # Limit content length 
    """

    try:
        # Query local LLM using Ollama
        response = ollama.generate(model="mistral", prompt=prompt)

        # Process LLM response
        llm_response = response['response']

        # Extract product info from LLM response
        if "No new product found" in llm_response:
            return None

        product_name = None
        product_description = None

        # Parse LLM output
        if "New Product Name:" in llm_response:
            product_name_start = llm_response.find("New Product Name:") + len("New Product Name:")
            product_name_end = llm_response.find("\n", product_name_start)
            if product_name_end == -1:
                product_name_end = len(llm_response)
            product_name = llm_response[product_name_start:product_name_end].strip()

        if "Product Description:" in llm_response:
            desc_start = llm_response.find("Product Description:") + len("Product Description:")
            desc_end = llm_response.find("\n", desc_start)
            if desc_end == -1:
                desc_end = len(llm_response)
            product_description = llm_response[desc_start:desc_end].strip()

            # Ensure description is under 180 characters
            if product_description and len(product_description) > 180:
                product_description = product_description[:177] + "..."

        if product_name:
            return {
                "new_product": product_name,
                "product_description": product_description or ""
            }
        return None

    except Exception as e:
        print(f"Error extracting product info using LLM: {e}")
        return None

    # Main execution


try:
    # Fetch company data
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers={'User-Agent': 'as872832@ucf.edu'})
    response.raise_for_status()
    data = response.json()

    # Convert the JSON data to a Pandas DataFrame
    company_data = pd.DataFrame.from_dict(data, orient='index')

    # Convert 'cik_str' column to string and pad with leading zeros
    company_data['cik_str'] = company_data['cik_str'].astype(str).str.zfill(10)

    # Limit to first 100 companies (or fewer) as specified in the instructions
    company_data = company_data.head(100)

    # Process each company
    for index, row in company_data.iterrows():
        cik = row['cik_str']
        ticker = row['ticker']
        company_name = row['title']
        print(f"Processing {company_name} (CIK: {cik}, Ticker: {ticker})...")

        # Get 8-K filings for this company
        filings = get_8k_filings(cik)

        for filing_date, accession_number in filings[:5]:  # Limit to 5 most recent filings per company
            print(f"  Analyzing filing from {filing_date}...")

            # Get filing content
            content = get_filing_content(accession_number, cik)

            # Extract product information
            product_info = extract_product_info(content, company_name, ticker)

            # If product information was found, add to results
            if product_info:
                results_df = pd.concat([results_df, pd.DataFrame([{
                    'company_name': company_name,
                    'stock_name': ticker,
                    'filing_time': filing_date,
                    'new_product': product_info['new_product'],
                    'product_description': product_info['product_description']
                }])], ignore_index=True)
                print(f"  Found new product: {product_info['new_product']}")

                # Add delay to avoid overwhelming the SEC API
            time.sleep(1)

            # Save results to CSV
    results_df.to_csv('sec_8k_product_releases.csv', index=False)
    print(f"Analysis complete. Found {len(results_df)} products. Results saved to sec_8k_product_releases.csv")

except requests.exceptions.RequestException as e:
    print(f"An error occurred with the request: {e}")
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")