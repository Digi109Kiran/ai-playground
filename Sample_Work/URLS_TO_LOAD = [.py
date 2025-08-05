URLS_TO_LOAD = [
    "https://docs.digital.ai/release/docs/category/get-started",
    "https://docs.digital.ai/release/docs/concept/core-concepts-of-xl-release",
    "https://docs.digital.ai/release/docs/glossary/release-glossary",
    "https://docs.digital.ai/release/docs/how-to/using-the-release-overview",
    "https://docs.digital.ai/release/docs/concept/release-life-cycle",
    "https://docs.digital.ai/release/docs/how-to/configure-release-properties",
    "https://docs.digital.ai/release/docs/category/phases",
    "https://docs.digital.ai/release/docs/category/tasks",
    "https://docs.digital.ai/release/docs/category/triggers"
]


# Step 1: Load and clean content from URLs
def load_url_documents(url_list):
    url_docs = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for url in url_list:
        try:
            print(f"🌐 Scraping: {url}")
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract only visible text
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n", strip=True)

            url_docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"❌ Error loading {url}: {e}")
    return url_docs