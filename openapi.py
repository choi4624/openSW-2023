import requests

url = 'http://apis.data.go.kr/1741000/DisasterMsg3/getDisasterMsg1List?'

serviceKey="H2Ez6uSzOlBqdDHho6M38bYByaSYK45E%2FTjqS%2BP5o9vFYCSXtHWbYurT6OajsxzD4q4I9t%2Bd0XAg63PFCZbmrQ%3D%3D"
params ={'serviceKey' : serviceKey, 'pageNo' : '1', 'numOfRows' : '10', 'type' : 'json' }

response = requests.get(url, params=params)
print(response.url)
print(response.content)