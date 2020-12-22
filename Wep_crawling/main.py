import requests
from bs4 import BeautifulSoup

# 새로운 함수
def crawling(soup):
# soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    result = []
    tbody = soup.find("tbody")
    for p in tbody.find_all("p", class_ = "title"):
        result.append(p.get_text().replace('\n','').strip())

    # result = []
    # tbody = soup.find("tbody")  # 실시간 차트
    # for div in tbody.find_next("div", class_="ellipsis rank01"):
    #    result.append(div.get_text().replace('\n', ''))

    return result


def main():
    custom_header = {
        'referer': 'https://https://music.bugs.co.kr/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Whale/2.8.108.15 Safari/537.36'
    }
    url = "https://music.bugs.co.kr/chart"
    req = requests.get(url, headers=custom_header)
    soup = BeautifulSoup(req.text, "html.parser")
    # crawling 함수의 결과를 출력합니다.

    print(crawling(soup))

if __name__ == "__main__":
        main()