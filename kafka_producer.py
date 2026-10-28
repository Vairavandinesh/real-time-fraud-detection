
import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda data: json.dumps(data).encode("utf-8"),
)

transactions = [
    {
        "transaction_id": f"TXN{i:03}",
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "type": txn_type,
    }
    for i, (amount, old_org, new_org, old_dest, new_dest, txn_type) in enumerate([
        (9000, 10000, 1000, 5000, 14000, "TRANSFER"),
        (200, 1000, 800, 2000, 2200, "PAYMENT"),
        (15000, 15000, 0, 1000, 16000, "CASH_OUT"),
        (500, 5000, 4500, 3000, 3500, "DEBIT"),
        (3000, 8000, 5000, 2000, 5000, "TRANSFER"),
    ], start=1)
]

try:
    for transaction in transactions:
        producer.send("transactions", value=transaction).get(timeout=10)
        print(f"Sent: {transaction['transaction_id']}")
        time.sleep(1)
finally:
    producer.flush()
    producer.close()

print("Finished sending 5 transactions.")
