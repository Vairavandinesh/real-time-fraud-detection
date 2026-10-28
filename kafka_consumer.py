
import json
import csv
import os
import requests
from kafka import KafkaConsumer

API_URL = "http://127.0.0.1:8000/predict"
OUTPUT_FILE = "fraud_predictions.csv"

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-detection-group",
    value_deserializer=lambda data: json.loads(data.decode("utf-8")),
)

print("Waiting for transactions...")

try:
    for message in consumer:
        transaction = message.value.copy()
        transaction_id = transaction.pop("transaction_id", "UNKNOWN")

        try:
            response = requests.post(
                API_URL,
                json=transaction,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            prediction = result["prediction"]
            probability = result["fraud_probability"]

            # Append the result to CSV
            file_exists = os.path.exists(OUTPUT_FILE)
            write_header = (
                not file_exists or os.path.getsize(OUTPUT_FILE) == 0
            )

            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "transaction_id",
                        "amount",
                        "type",
                        "prediction",
                        "fraud_probability",
                    ],
                )

                if write_header:
                    writer.writeheader()

                writer.writerow({
                    "transaction_id": transaction_id,
                    "amount": transaction["amount"],
                    "type": transaction["type"],
                    "prediction": prediction,
                    "fraud_probability": probability,
                })

            print(
                f"{transaction_id} | {prediction} | "
                f"Fraud probability: {probability}"
            )

        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"Failed to process {transaction_id}: {error}")

except KeyboardInterrupt:
    print("\nConsumer stopped.")

finally:
    consumer.close()
