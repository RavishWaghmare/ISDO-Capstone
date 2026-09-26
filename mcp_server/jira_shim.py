"""
ISDO Lab C2 — Mock Jira Service Management REST API (Flask Shim)
Mimics the Jira REST API for service requests so the MCP server
can make real HTTP calls without touching production.

Endpoints:
  GET  /rest/agile/1.0/board/requests    — list all requests
  GET  /rest/api/2/issue/<key>           — get one request
  PUT  /rest/api/2/issue/<key>           — update a request
  GET  /rest/api/2/issue?request_type=Access+Grant — filter
  POST /rest/api/2/issue                 — create a request

Run with:  python jira_shim.py
Default port: 5002
"""

from flask import Flask, jsonify, request
import csv
import os

app = Flask(__name__)

DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "requests.csv"))

def load_requests():
    requests_data = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                if None in row:  # extra columns = unquoted comma in the CSV
                    print(f"Warning: skipping malformed row {line_no} in {DATA_FILE} "
                          f"(too many columns - quote fields that contain commas)")
                    continue
                requests_data[row["key"]] = dict(row)
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return requests_data

REQUESTS = load_requests()

@app.route("/rest/agile/1.0/board/requests", methods=["GET"])
@app.route("/rest/api/2/issue", methods=["GET"])
def list_requests():
    """Return all service requests, optionally filtered."""
    results = list(REQUESTS.values())

    for key in ["request_type", "priority", "assignee", "status"]:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]

    return jsonify({"issues": results, "total": len(results)})

@app.route("/rest/api/2/issue/<key>", methods=["GET"])
def get_request(key):
    """Return a single request by key."""
    req = REQUESTS.get(key)
    if not req:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    # Mimic Jira's nested fields structure
    return jsonify({
        "key": key,
        "fields": {
            "summary": req.get("summary"),
            "priority": {"name": req.get("priority")},
            "status": {"name": req.get("status")},
            "assignee": {"displayName": req.get("assignee")},
            "customfield_sla": req.get("sla"),
            "issuetype": {"name": req.get("request_type")},
        }
    })

@app.route("/rest/api/2/issue/<key>", methods=["PUT"])
def update_request(key):
    """Update a service request."""
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"errorMessages": ["No update body provided"]}), 400

    # Accept Jira format {"fields": {...}} and flatten {"status": {"name": "Done"}} -> "Done"
    fields = data.get("fields", data)
    flat = {k: (v.get("name") or v.get("displayName")) if isinstance(v, dict) else v
            for k, v in fields.items() if k != "key"}
    REQUESTS[key].update(flat)
    fields = flat
    print(f"[Jira Mock] Updated {key}: {fields}")
    return jsonify({"key": key, "message": "Updated successfully"})

@app.route("/rest/api/2/issue", methods=["POST"])
def create_request():
    """Create a new service request."""
    data = request.get_json(silent=True) or {}
    fields = data.get("fields", {})
    if not fields.get("summary"):
        return jsonify({"errorMessages": ["Field 'summary' is required"]}), 400
    # Next key after the highest existing one (REQ-1010 -> REQ-1011)
    next_num = max([int(k.split("-")[1]) for k in REQUESTS] or [1000]) + 1
    key = f"REQ-{next_num}"
    REQUESTS[key] = {
        "key": key,
        "summary": fields.get("summary", ""),
        "request_type": fields.get("issuetype", {}).get("name", ""),
        "priority": fields.get("priority", {}).get("name", "Medium"),
        "assignee": "",
        "sla": "",
        "status": "Open"
    }
    print(f"[Jira Mock] Created request: {key}")
    return jsonify({"key": key, "message": "Request created"}), 201

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})

if __name__ == "__main__":
    print(f"Jira Mock API starting on http://localhost:5002")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    print("Endpoints: GET /rest/agile/1.0/board/requests  |  GET /health")
    # use_reloader=False: the auto-reloader would wipe in-memory updates on every code save
    app.run(port=5002, debug=True, use_reloader=False)
