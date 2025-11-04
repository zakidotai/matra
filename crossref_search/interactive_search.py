import pandas as pd
import requests
import time
import os
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ CONFIG ============ #
email = 'mzaki4@jh.edu'  # Replace with your email
max_rows = 1000
output_base = './dois_crossref'
# ================================= #

def cross_reference_search_all_journals(query, start_date, end_date, email, issn=None):
    """
    Search across all journals or a specific ISSN for a given date range
    """
    cursor = "*"
    keep_paging = True
    
    base_url = 'https://api.crossref.org/works?query='
    headers = {'Accept': 'application/json', 'mailto': email}
    
    # Build filters
    filters = [f'from-pub-date:{start_date}', f'until-pub-date:{end_date}']
    if issn:
        filters.append(f'issn:{issn}')
    
    params = {'filter': ",".join(filters)}
    results = []
    
    while keep_paging:
        try:
            print(f"Searching for '{query}' from {start_date} to {end_date} (ISSN={issn if issn else 'ALL'})")
            r = requests.get(
                base_url + query + "&rows=" + str(max_rows) + "&cursor=" + cursor,
                headers=headers,
                timeout=100,
                params=params
            )
            data = r.json()
            cursor = quote(data['message']['next-cursor'], safe='')
            
            items = data['message']['items']
            if not items:
                keep_paging = False
                break
            
            for item in items:
                journal = item.get('container-title', ['None'])[0]
                issn_val = item.get('ISSN', ['None'])[0]
                title = item.get('title', ['None'])[0] if 'title' in item and item['title'] else 'None'
                
                result = {
                    'DOI': item['DOI'],
                    'Query': query,
                    'PII': 'None',
                    'Title': title,
                    'Journal': journal,
                    'ISSN': issn_val,
                    'Date_Range': f"{start_date}_{end_date}"
                }
                results.append(result)
                
                if len(results) % 500 == 0:
                    print(f'Papers found so far: {len(results)}')
                    time.sleep(1)
        except Exception as e:
            print(f"Error occurred: {e}")
            keep_paging = False
    
    return results


def run_yearly_search(query, start_year, end_year, merge=False, issn=None):
    """
    Run yearly searches in parallel and save results
    """
    # Create output folder
    query2 = query.replace(' ', '_')
    issn_tag = f"_issn_{issn}" if issn else "_all_journals"
    output_dir = f"{output_base}_{query2}{issn_tag}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Make yearly ranges
    year_ranges = [(f"{year}-01", f"{year+1}-01") for year in range(start_year, end_year)]
    
    all_results = []
    
    # Parallel execution
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(cross_reference_search_all_journals, f'"{query}"', s, e, email, issn): (s, e) 
                   for s, e in year_ranges}
        
        for future in as_completed(futures):
            s, e = futures[future]
            try:
                results = future.result()
                if results:
                    TempDF = pd.DataFrame(results).set_index('DOI')
                    filename = f'all_journals_{s.split("-")[0]}.csv'
                    TempDF.to_csv(os.path.join(output_dir, filename))
                    print(f"Saved {len(TempDF)} results for {s} to {e}")
                    all_results.append(TempDF)
                else:
                    print(f"No results found for {s} to {e}")
            except Exception as exc:
                print(f"Error for {s}-{e}: {exc}")
    
    # Merge all results
    if merge and all_results:
        merged = pd.concat(all_results)
        merged_file = os.path.join(output_dir, f'merged_{start_year}_{end_year}.csv')
        merged.to_csv(merged_file)
        print(f"✅ Merged file saved as {merged_file}")


if __name__ == "__main__":
    # ===== Ask user for inputs ===== #
    query = input("Enter search query: ").strip()
    start = input("Enter start year-month (YYYY-MM): ").strip()
    end = input("Enter end year-month (YYYY-MM): ").strip()
    issn = input("Enter journal ISSN (optional, leave blank for all): ").strip() or None
    merge_flag = input("Merge all results after run? (yes/no): ").strip().lower() == "yes"
    
    start_year = int(start.split("-")[0])
    end_year = int(end.split("-")[0])
    
    run_yearly_search(query, start_year, end_year, merge=merge_flag, issn=issn)
