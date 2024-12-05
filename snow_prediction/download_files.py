import requests
from tqdm import tqdm

from snow_prediction.html_parser import ImisHTMLParser


def download_file(url: str):
    response = requests.get(url)
    components = url.split("/")
    name = components[len(components) - 1]
    with open(f"data/{name}", "wb") as file:
        file.write(response.content)

url = "https://measurement-data.slf.ch/imis/data/by_station/"
parser = ImisHTMLParser(url)
urls = parser.parse_html()

for url in tqdm(urls):
    download_file(url)


