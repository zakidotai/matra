import os
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========= CONFIG ========= #
email = 'mzaki4@jh.edu'
crossref_dir = '../crossref_search/dois_crossref_carbides_Ti_issn_0925-8388/'   # Folder containing CSVs with DOIs
keys = list(pd.read_csv('keys.csv').keys_list.values)    # Elsevier API keys
output_base = './corpus'                                 # Where XMLs are saved
max_workers = 10                                         # Parallel threads
# ========================== #


def already_downloaded(doi, journal):
    """
    Check if DOI already exists in corpus folder (by PII.xml).
    """
    jdir = "_".join(journal.split())
    output_dir = os.path.join(output_base, jdir)
    if not os.path.exists(output_dir):
        return False

    for pii_folder in os.listdir(output_dir):
        xml_path = os.path.join(output_dir, pii_folder, f"{pii_folder}.xml")
        if os.path.exists(xml_path):
            try:
                with open(xml_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                doi_tag = soup.find("doi")
                if doi_tag and doi_tag.text.lower() == doi.lower():
                    return True
            except:
                continue
    return False


def article_downloader(doi, journal, keys, output_base):
    """
    Download XML from DOI (Elsevier API).
    Saves to ./corpus/{Journal}/{PII}/{PII}.xml
    """
    try:
        # Skip if already downloaded
        if already_downloaded(doi, journal):
            return f"⏭️ Skipped (already downloaded): {doi}"

        jdir = "_".join(journal.split())
        output_dir = os.path.join(output_base, jdir)
        os.makedirs(output_dir, exist_ok=True)

        # Pick random API key
        api_id = random.randint(0, len(keys)-1)
        xml_url = f"https://api.elsevier.com/content/article/doi/{doi}?APIKey={keys[api_id]}"
        headers = {"User-Agent": "Mozilla/5.0"}
        url_get = requests.get(xml_url, headers=headers, timeout=60)

        soup = BeautifulSoup(url_get.content, "lxml")
        pii_tag = soup.find("xocs:pii-unformatted")
        if not pii_tag:
            with open("err.txt", "a") as rep:
                rep.write(f"{doi}\n")
            return f"❌ Failed (no PII) for {doi}"

        pii = pii_tag.text
        soup_path = os.path.join(output_dir, str(pii))
        os.makedirs(soup_path, exist_ok=True)
        xmlpath = os.path.join(soup_path, f"{pii}.xml")

        with open(xmlpath, "w", encoding="utf-8") as file:
            file.write(str(soup))

        return f"✅ Downloaded {doi} -> {xmlpath}"

    except Exception as e:
        with open("err.txt", "a") as rep:
            rep.write(f"{doi}\n")
        return f"⚠️ Error {doi}: {e}"


def process_csvs(crossref_dir, keys, output_base, max_workers=10):
    """
    Process all CSV files, deduplicate DOIs, download in parallel.
    """
    # Load all DOIs from all CSVs
    all_dfs = []
    for f in os.listdir(crossref_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(crossref_dir, f))
            if "DOI" in df.columns and "Journal" in df.columns:
                all_dfs.append(df)
    masterdf = pd.concat(all_dfs).drop_duplicates(subset=["DOI"])
    masterdf["Journal"] = masterdf["Journal"].astype(str).str.replace(" &amp; ", " and")
    masterdf = masterdf.reset_index(drop=True)

    print(f"📊 Total unique DOIs: {len(masterdf)}")

    # Parallel downloads
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in masterdf.iterrows():
            doi, journal = row["DOI"], row["Journal"]
            tasks.append(executor.submit(article_downloader, doi, journal, keys, output_base))

        for future in as_completed(tasks):
            print(future.result())


if __name__ == "__main__":
    process_csvs(crossref_dir, keys, output_base, max_workers=max_workers)

# def article_downloader(doi, journal, keys, output_base):
#     """
#     Download XML from DOI (Elsevier API).
#     Saves to ./corpus/{Journal}/{PII}/{PII}.xml
#     """
#     try:
#         jdir = "_".join(journal.split())
#         output_dir = os.path.join(output_base, jdir)
#         os.makedirs(output_dir, exist_ok=True)

#         # Pick random API key
#         api_id = random.randint(0, len(keys)-1)
#         xml_url = f"https://api.elsevier.com/content/article/doi/{doi}?APIKey={keys[api_id]}"
#         headers = {"User-Agent": "Mozilla/5.0"}
#         url_get = requests.get(xml_url, headers=headers, timeout=60)

#         soup = BeautifulSoup(url_get.content, "lxml")
#         pii_tag = soup.find("xocs:pii-unformatted")
#         if not pii_tag:
#             with open("err.txt", "a") as rep:
#                 rep.write(f"{doi}\n")
#             return f"❌ Failed (no PII) for {doi}"

#         pii = pii_tag.text
#         soup_path = os.path.join(output_dir, str(pii))
#         os.makedirs(soup_path, exist_ok=True)
#         xmlpath = os.path.join(soup_path, f"{pii}.xml")

#         with open(xmlpath, "w", encoding="utf-8") as file:
#             file.write(str(soup))

#         return f"✅ Downloaded {doi} -> {xmlpath}"

#     except Exception as e:
#         with open("err.txt", "a") as rep:
#             rep.write(f"{doi}\n")
#         return f"⚠️ Error {doi}: {e}"


# def process_csv(csv_path, keys, output_base, max_workers=10):
#     """
#     Process one CSV file (download all DOIs inside).
#     """
#     print(f"\n📂 Processing {csv_path}")
#     df = pd.read_csv(csv_path)
#     if "DOI" not in df.columns or "Journal" not in df.columns:
#         print(f"Skipping {csv_path} (missing DOI/Journal columns)")
#         return

#     # Fix journal names
#     df["Journal"] = df["Journal"].astype(str).str.replace(" &amp; ", " and")

#     tasks = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for _, row in df.iterrows():
#             doi = row["DOI"]
#             journal = row["Journal"]
#             tasks.append(executor.submit(article_downloader, doi, journal, keys, output_base))

#         for future in as_completed(tasks):
#             print(future.result())


# if __name__ == "__main__":
#     # Loop through all Crossref CSVs
#     jlist = [os.path.join(crossref_dir, f) for f in os.listdir(crossref_dir) if f.endswith(".csv")]

#     for csv_file in jlist:
#         process_csv(csv_file, keys, output_base, max_workers=max_workers)
