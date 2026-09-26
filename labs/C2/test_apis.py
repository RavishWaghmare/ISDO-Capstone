"""
ISDO Lab C2 - Verify both mock APIs (Steps 3-6 + CRUD round-trip)

Start both shims first (two terminals, from the project root):
    python mcp_server/snow_shim.py
    python mcp_server/jira_shim.py
Then run:
    python labs/C2/test_apis.py
"""

import json
import sys

import requests

SNOW = "http://localhost:5001"
JIRA = "http://localhost:5002"
passed, failed = 0, 0


def check(label, ok, detail=""):
    global passed, failed
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -> {detail}" if detail else ""))


def get(url, **kw):
    r = requests.get(url, timeout=5, **kw)
    return r.status_code, r.json()


def main():
    # Step 6 first: if a shim is down, say so clearly instead of a stack trace
    print("\n--- Step 6: health checks ---")
    for name, base in [("ServiceNow", SNOW), ("Jira", JIRA)]:
        try:
            code, body = get(f"{base}/health")
            check(f"{name} /health", code == 200 and body.get("status") == "ok", json.dumps(body))
        except requests.ConnectionError:
            check(f"{name} /health", False, f"not reachable at {base} - is the shim running?")
    if failed:
        sys.exit(1)

    print("\n--- Step 3: ServiceNow list ---")
    code, body = get(f"{SNOW}/api/now/table/incident")
    check("GET /api/now/table/incident returns 15", body["total"] == 15, f"total={body['total']}")

    print("\n--- Step 4: ServiceNow filters ---")
    code, body = get(f"{SNOW}/api/now/table/incident", params={"priority": "P1"})
    nums = [r["number"] for r in body["result"]]
    check("?priority=P1 only returns P1", all(r["priority"] == "P1" for r in body["result"]),
          f"total={body['total']} {nums}")
    code, body = get(f"{SNOW}/api/now/table/incident", params={"category": "Network"})
    check("?category=Network", body["total"] > 0 and all(r["category"] == "Network" for r in body["result"]),
          f"total={body['total']}")
    code, body = get(f"{SNOW}/api/now/table/incident/INC0001001")
    check("GET /incident/INC0001001", code == 200 and body["result"]["number"] == "INC0001001",
          body["result"]["short_description"])
    code, _ = get(f"{SNOW}/api/now/table/incident/INC9999999")
    check("Unknown incident returns 404", code == 404)

    print("\n--- ServiceNow PATCH round-trip (used by SLA Agent in C5) ---")
    r = requests.patch(f"{SNOW}/api/now/table/incident/INC0001008",
                       json={"state": "Escalated", "work_notes": "Lab C2 test"}, timeout=5)
    check("PATCH state=Escalated", r.status_code == 200)
    code, body = get(f"{SNOW}/api/now/table/incident/INC0001008")
    check("Read back shows Escalated", body["result"]["state"] == "Escalated", body["result"]["state"])
    requests.patch(f"{SNOW}/api/now/table/incident/INC0001008", json={"state": "Open"}, timeout=5)

    print("\n--- Step 5: Jira list + filters ---")
    code, body = get(f"{JIRA}/rest/agile/1.0/board/requests")
    check("GET /rest/agile/1.0/board/requests returns 10", body["total"] == 10, f"total={body['total']}")
    code, body = get(f"{JIRA}/rest/api/2/issue", params={"request_type": "Access Grant"})
    check("?request_type=Access Grant", body["total"] == 2, f"{[i['key'] for i in body['issues']]}")

    print("\n--- Jira single issue (nested 'fields') ---")
    code, body = get(f"{JIRA}/rest/api/2/issue/REQ-1002")
    ok = code == 200 and body["fields"]["priority"]["name"] == "High"
    check("GET /rest/api/2/issue/REQ-1002", ok, body["fields"]["summary"])
    code, _ = get(f"{JIRA}/rest/api/2/issue/REQ-9999")
    check("Unknown issue returns 404", code == 404)

    print("\n--- Jira PUT round-trip ---")
    r = requests.put(f"{JIRA}/rest/api/2/issue/REQ-1002",
                     json={"fields": {"status": {"name": "In Progress"}}}, timeout=5)
    check("PUT status=In Progress", r.status_code == 200)
    code, body = get(f"{JIRA}/rest/api/2/issue/REQ-1002")
    check("Read back shows In Progress", body["fields"]["status"]["name"] == "In Progress",
          body["fields"]["status"]["name"])
    requests.put(f"{JIRA}/rest/api/2/issue/REQ-1002", json={"status": "Open"}, timeout=5)

    print(f"\nResult: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
