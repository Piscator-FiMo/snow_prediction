from html.parser import HTMLParser

import requests


class ImisHTMLParser(HTMLParser):
    links: list = []
    url: str = ""
    def __init__(self, url):
        super().__init__()
        self.url = url

    def parse_html(self) -> list:
        response = requests.get(self.url)
        self.feed(response.text)
        return self.links


    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[0] == 'href':
                    if attr[1].endswith('csv'):
                        self.links.append('https://measurement-data.slf.ch' + attr[1])
