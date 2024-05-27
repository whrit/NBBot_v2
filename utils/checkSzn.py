import requests

url = "https://api-nba-v1.p.rapidapi.com/seasons"

headers = {
	"X-RapidAPI-Key": "04bbc1c2d6mshbeed6a20bcbb5edp11712bjsne4f585ea9eb4",
	"X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())