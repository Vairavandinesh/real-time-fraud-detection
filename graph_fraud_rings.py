import pandas as pd
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "dataset.csv"
OUTPUT_FILE = BASE_DIR / "graph_fraud_rings.csv"

SAMPLE_SIZE = 50000

print("Loading a sample of transactions...")
df = pd.read_csv(
    DATA_FILE,
    usecols=["nameOrig", "nameDest", "amount", "isFraud"],
    nrows=SAMPLE_SIZE
).dropna(subset=["nameOrig", "nameDest"])

df["isFraud"] = df["isFraud"].astype(int)

print(f"Transactions loaded: {len(df):,}")
print("Building account graph...")

graph = nx.DiGraph()

for row in df.itertuples(index=False):
    graph.add_edge(
        row.nameOrig,
        row.nameDest,
        amount=float(row.amount),
        is_fraud=int(row.isFraud)
    )

print(f"Accounts: {graph.number_of_nodes():,}")
print(f"Transactions (edges): {graph.number_of_edges():,}")

# Rank accounts by transaction activity and links to known fraud.
account_stats = {}

for source, destination, attributes in graph.edges(data=True):
    fraud = attributes["is_fraud"]

    for account in (source, destination):
        if account not in account_stats:
            account_stats[account] = {
                "account_id": account,
                "transactions": 0,
                "fraud_linked_transactions": 0,
                "total_amount": 0.0,
            }

        stats = account_stats[account]
        stats["transactions"] += 1
        stats["fraud_linked_transactions"] += fraud
        stats["total_amount"] += attributes["amount"]

results = pd.DataFrame(account_stats.values())

results["risk_score"] = (
    results["fraud_linked_transactions"]
    / results["transactions"]
)

results = results.sort_values(
    ["fraud_linked_transactions", "transactions"],
    ascending=False
)

results.to_csv(OUTPUT_FILE, index=False)

print("\nTop 10 accounts connected to fraud-labelled transactions:")
print(results.head(10).to_string(index=False))
print(f"\nResults saved to: {OUTPUT_FILE}")
print("\nNote: This is a graph-based screening heuristic, not a trained GNN.")
