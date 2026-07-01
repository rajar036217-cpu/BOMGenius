import json
import os

LOCAL_RULES_FILE = "federated/global_rules.json"

def log_human_feedback(feedback: dict):

    ebom_part = feedback["ebom_part"]
    correct_part = feedback["correct_part"]

    rules = {}

    if os.path.exists(LOCAL_RULES_FILE):
        rules = json.load(open(LOCAL_RULES_FILE))

    rules[ebom_part] = correct_part

    with open(LOCAL_RULES_FILE, "w") as f:
        json.dump(rules, f, indent=2)



def export_local_updates():
    if not os.path.exists(LOCAL_RULES_FILE):
        return []
    return json.load(open(LOCAL_RULES_FILE))