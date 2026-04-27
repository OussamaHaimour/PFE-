from kafka import KafkaConsumer
import requests
import json

consumer = KafkaConsumer(
    'helpdesk_emails',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

API_URL = "http://127.0.0.1:5000/predict"

for message in consumer:
    data = message.value
    response = requests.post(API_URL, json=data)
    print(response.json())