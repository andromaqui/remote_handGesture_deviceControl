from decouple import config
import requests

url="https://api.lifx.com/v1/lights/all/state"
token = config('API_TOKEN')
headers = {
    "Authorization": "Bearer %s" % token,
}

##sends a request to lifx api
##and turns on the smart bulb
def turn_on():
    payload = {
        "power": "on",
        }
    response = requests.put(url, data=payload, headers=headers)
    print(response)



##sends a request to lifx api
##and turns off the smart bulb
def turn_off():
    payload = {
        "power": "off",
        }
    response = requests.put(url, data=payload, headers=headers)
    print(response)

turn_off()