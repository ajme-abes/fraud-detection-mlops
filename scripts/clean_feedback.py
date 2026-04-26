import json

path = "logs/feedback.jsonl"

with open(path) as f:
    entries = [json.loads(line) for line in f]

valid = [e for e in entries if e["actual_label"] in (0, 1)]
invalid = [e for e in entries if e["actual_label"] not in (0, 1)]

print(f"Total: {len(entries)}, Valid: {len(valid)}, Removed: {len(invalid)}")
for e in invalid:
    print(f"  Removed: actual_label={e['actual_label']} id={e['request_id'][:16]}...")

with open(path, "w") as f:
    for e in valid:
        f.write(json.dumps(e) + "\n")

print("feedback.jsonl cleaned.")
