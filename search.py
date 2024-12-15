import os
from datetime import datetime, timedelta

import requests
import serpapi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()


class Search:
    def __init__(self):
        self.wiki_url = "https://en.wikipedia.org/w/api.php"
        self.newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

    def wikipedia(self, query) -> list:
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }

        response = requests.get(self.wiki_url, params=search_params)
        data = response.json()

        title = data["query"]["search"][0]["title"]

        content_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        }

        response = requests.get(self.wiki_url, params=content_params)
        data = response.json()

        page_id = next(iter(data["query"]["pages"]))

        content = data["query"]["pages"][page_id]["extract"]

        url = f"https://en.wikipedia.org/?curid={page_id}"

        return (content, url)

    def news(self, query):
        return self.newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            from_param=(datetime.now() - timedelta(2)).strftime("%Y-%m-%d"),
            to=datetime.today().strftime("%Y-%m-%d"),
            page_size=7,
        )

    def google(self, query) -> dict:
        params = {
            "q": query,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com",
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "k": 5,
        }

        search = serpapi.search(params)
        results = search.as_dict()
        results.pop("search_metadata")
        results.pop("search_parameters")
        results.pop("search_information")
        results.pop("pagination")
        results.pop("serpapi_pagination")
        try:
            results.pop("related_questions")
        except:
            pass
        return results

    def fetch_url(self, url) -> dict:
        # Page content from Website URL
        page = requests.get(url)

        # Function to remove tags
        def remove_tags(html):
            # parse html content
            soup = BeautifulSoup(html, "html.parser")

            for data in soup(["style", "script", "head", "title"]):
                # Remove tags
                data.decompose()

            # return data by retrieving the tag content
            return " ".join(soup.stripped_strings)

        # Print the extracted data
        return remove_tags(page.content)
        # return [remove_tags(page.content).encode('utf-8')]
