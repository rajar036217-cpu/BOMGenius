import requests
from collections import Counter
import json

FACTORY_NODES = [
    "http://127.0.0.1:8000",  # Factory A
    "http://127.0.0.1:8001",  # Factory B (if you simulate)
]


def aggregate():
    all_updates = []

    for node in FACTORY_NODES:
        r = requests.get(f"{node}/federated/export")
        all_updates.append(r.json())

    make_buy_votes = {}

    for node_updates in all_updates:
        for u in node_updates:
            pn = u["part_no"]
            make_buy_votes.setdefault(pn, []).append(u.get("correct_make_buy"))

    global_rules = {}
    for pn, votes in make_buy_votes.items():
        global_rules[pn] = {"Make_Buy": Counter(votes).most_common(1)[0][0]}

    print("Aggregated Rules:", global_rules)

    for node in FACTORY_NODES:
        requests.post(f"{node}/federated/import", json=global_rules)


if __name__ == "__main__":
    aggregate()
