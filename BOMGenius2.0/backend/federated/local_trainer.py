import json
import os

LOCAL_UPDATES_FILE = "federated/local_updates.json"


def log_human_feedback(feedback: dict):
    """
    feedback example:
    {
      "part_no": "CPU-778",
      "correct_make_buy": "BUY",
      "correct_uom": "EA",
      "correct_work_center": "Assembly"
    }
    """
    updates = []
    if os.path.exists(LOCAL_UPDATES_FILE):
        updates = json.load(open(LOCAL_UPDATES_FILE))

    updates.append(feedback)
    with open(LOCAL_UPDATES_FILE, "w") as f:
        json.dump(updates, f, indent=2)


def export_local_updates():
    if not os.path.exists(LOCAL_UPDATES_FILE):
        return []
    return json.load(open(LOCAL_UPDATES_FILE))
