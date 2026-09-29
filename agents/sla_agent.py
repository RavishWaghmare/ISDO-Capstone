"""
ISDO Lab C5 - SLA & Escalation Agent
Checks SLA breach risk, escalates CRITICAL/BREACHED P1/P2 tickets, and pauses at a
human-in-the-loop (HITL) gate before ANY P1 escalation.

Run from the project root (start mcp_server/snow_shim.py first to see real PATCH calls):
    python agents/sla_agent.py
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNOW_URL = "http://localhost:5001/api/now/table/incident"   # Lab C2 ServiceNow shim
AUDIT_LOG = PROJECT_ROOT / "logs" / "hitl_audit.log"
MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
MAX_TURNS = 6

NOW = datetime(2024, 1, 15, 10, 30)                          # simulated "now" (Step 1)
SLA_TARGET_MIN = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
ESCALATE_RISKS, ESCALATE_PRIORITIES = {"BREACHED", "CRITICAL"}, {"P1", "P2"}
REJECTED = set()   # tickets whose P1 escalation a human already declined this run
TEAMS = ["L2-Network-Ops", "L2-App-Support", "L2-Server-Ops", "L2-Security-Ops", "L2-Service-Desk"]

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set - add it to the .env file in the project root.")
client = anthropic.Anthropic()

# -- TOOLS ---------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_sla_status",
        "description": "Check a ticket's SLA: minutes remaining, breach_risk "
                       "(BREACHED/CRITICAL/AT_RISK/ON_TRACK) and whether escalation is required.",
        "input_schema": {"type": "object", "properties": {
            "ticket_number": {"type": "string"},
            "sla_due": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]}},
            "required": ["ticket_number", "sla_due", "priority"]},
    },
    {
        "name": "update_ticket",
        "description": "Update a ticket in ServiceNow: escalate it to an L2 team, add a work note, or change its state.",
        "input_schema": {"type": "object", "properties": {
            "ticket_number": {"type": "string"},
            "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
            "escalation_team": {"type": "string", "enum": TEAMS, "description": "Required for escalate"},
            "note": {"type": "string", "description": "Work note text (for add_note, optional for escalate)"},
            "new_state": {"type": "string", "enum": ["In Progress", "On Hold", "Resolved"],
                          "description": "Required for update_state"}},
            "required": ["ticket_number", "action"]},
    },
]

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.
For each ticket:
1. Call get_sla_status once.
2. If requires_escalation is true, call update_ticket with action=escalate and the right team:
   Network -> L2-Network-Ops, Application -> L2-App-Support, Server -> L2-Server-Ops,
   Access/Security -> L2-Security-Ops, anything else -> L2-Service-Desk.
   Include a one-line note explaining the SLA risk.
3. If breach_risk is AT_RISK, add a work note (add_note) so the assigned team is warned. Do not escalate.
4. If ON_TRACK, take no action.
If an escalation is rejected by the human approver, do NOT retry it - add a work note recording
that escalation was declined, then stop. Finish with one short summary line."""

def get_sla_status(ticket_number, sla_due, priority):
    """Minutes remaining vs the priority's SLA target -> breach risk (Step 1 rules)."""
    try:
        due = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due '{sla_due}' - expected YYYY-MM-DD HH:MM:SS"}
    minutes = int((due - NOW).total_seconds() // 60)
    target = SLA_TARGET_MIN[priority]
    if minutes < 0:
        risk, msg = "BREACHED", f"SLA breached {abs(minutes)} minutes ago"
    elif minutes < target * 0.2:
        risk, msg = "CRITICAL", f"Only {minutes} minutes remaining -- breach imminent"
    elif minutes < target * 0.5:
        risk, msg = "AT_RISK", f"{minutes} minutes remaining ({minutes / target:.0%} of {target}-min target)"
    else:
        risk, msg = "ON_TRACK", f"{minutes} minutes remaining -- on track"
    return {"ticket_number": ticket_number, "priority": priority, "sla_due": sla_due,
            "minutes_remaining": minutes, "sla_target_minutes": target, "breach_risk": risk,
            "status_message": msg,
            "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES}

def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """PATCH the C2 ServiceNow shim; fall back to a local simulation if it isn't running."""
    if action == "escalate":
        body = {"state": "Escalated", "assignment_group": escalation_team, "work_notes": note or ""}
        label = f"ESCALATED {ticket_number} -> {escalation_team}"
    elif action == "add_note":
        body, label = {"work_notes": note or ""}, f"NOTE ADDED to {ticket_number}: {(note or '')[:60]}"
    else:
        body, label = {"state": new_state}, f"STATE CHANGED {ticket_number} -> {new_state}"
    try:
        r = requests.patch(f"{SNOW_URL}/{ticket_number}", json=body, timeout=3)
        r.raise_for_status()
        source = "servicenow-shim"
    except requests.RequestException:
        source = "simulated (shim not running)"
    print(f"  [ServiceNow Mock] {label}  ({source})")
    return {"ticket_number": ticket_number, "action": action, "success": True, "source": source,
            "updated_fields": body, "timestamp": datetime.now().isoformat(timespec="seconds")}

# -- HITL GATE -------------------------------------------------------------------
def audit(entry):
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")

def hitl_approve(ticket_number, action, detail):
    """Pause for a human decision. Anything other than 'y' (including Ctrl+Z/EOF) is a rejection."""
    print(f"\n  {'!!! ' * 6}\n  HITL APPROVAL REQUIRED\n  Ticket:  {ticket_number}\n"
          f"  Action:  {action}\n  Detail:  {detail}\n  {'!!! ' * 6}")
    try:
        approved = input("  Approve escalation? [y/n]: ").strip().lower() == "y"
    except EOFError:
        approved = False
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED - escalation cancelled'}")
    audit({"ticket": ticket_number, "action": action, "detail": detail,
           "decision": "APPROVED" if approved else "REJECTED"})
    return approved

def guarded_update(ticket, inp, sla):
    """Every update_ticket call passes through here. Policy is enforced in code, not by the model."""
    if inp["ticket_number"] != ticket["number"]:
        return {"success": False, "message": f"Blocked: this run may only update {ticket['number']}"}
    if inp["action"] == "escalate":
        if not (sla and sla["requires_escalation"]):
            risk = sla["breach_risk"] if sla else "unknown"
            return {"success": False, "message": f"Blocked by policy: {ticket['priority']} / {risk} does not "
                    "qualify for escalation (only CRITICAL/BREACHED P1/P2). Add a note instead."}
        if not inp.get("escalation_team"):
            return {"success": False, "message": "escalation_team is required for escalate"}
        if ticket["number"] in REJECTED:  # a human already said no - never ask twice
            return {"success": False, "message": "Escalation REJECTED by human approver. Do not retry."}
        if ticket["priority"] == "P1" and not hitl_approve(
                ticket["number"], "Escalate ticket", f"Escalate to {inp['escalation_team']} ({sla['status_message']})"):
            REJECTED.add(ticket["number"])
            return {"success": False, "message": "Escalation REJECTED by human approver. Do not retry."}
    return update_ticket(inp["ticket_number"], inp["action"], inp.get("escalation_team"),
                         inp.get("note"), inp.get("new_state"))

# -- AGENTIC LOOP ------------------------------------------------------------------
def monitor_ticket(ticket):
    """Run SLA monitoring for one ticket; returns an outcome dict (used by C6)."""
    print(f"\n{'=' * 55}\nSLA Check: {ticket['number']} | {ticket['priority']} | Category: {ticket['category']}\n{'=' * 55}")
    messages = [{"role": "user", "content": "Monitor SLA for this ticket and escalate if needed:\n\n" + "\n".join(
        f"{k.replace('_', ' ').title()}: {v}" for k, v in ticket.items())}]
    # Policy uses the ticket's REAL data, not whatever the model passes to get_sla_status.
    sla = get_sla_status(ticket["number"], ticket["sla_due"], ticket["priority"])
    outcome = {"ticket_number": ticket["number"], "breach_risk": sla.get("breach_risk"),
               "minutes_remaining": sla.get("minutes_remaining"), "escalated_to": None, "hitl": None}

    for _ in range(MAX_TURNS):
        # temperature is not accepted by SDK 1.8.0 - low effort instead (see C3 notes)
        response = client.messages.create(model=MODEL, max_tokens=1024, output_config={"effort": "low"},
                                          system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        if response.stop_reason != "tool_use":
            text = " ".join(b.text for b in response.content if b.type == "text").strip()
            if text:
                print(f"  Agent: {text}")
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "get_sla_status":
                result = get_sla_status(**block.input)
                print(f"  -> Risk Level: {result.get('breach_risk')}\n  -> Status:     {result.get('status_message')}")
            elif block.name == "update_ticket":
                result = guarded_update(ticket, block.input, sla)
                if block.input["action"] == "escalate":
                    if result.get("success"):
                        outcome["escalated_to"] = block.input.get("escalation_team")
                    if ticket["priority"] == "P1" and sla["requires_escalation"]:
                        outcome["hitl"] = "APPROVED" if result.get("success") else "REJECTED"
                if not result.get("success"):
                    print(f"  -> {result['message']}")
            else:
                result = {"error": f"Unknown tool: {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})
    else:
        print(f"  !! Stopped after {MAX_TURNS} turns")
    return outcome

# -- RUN ---------------------------------------------------------------------------
# sla_due values chosen so the 4 tickets hit 4 different states at simulated now = 10:30.
TEST_TICKETS = [
    # P1, 10 min left of a 60-min target (17%) -> CRITICAL -> HITL prompt: type 'y'
    {"number": "INC0001002", "description": "Cannot access ERP system - SAP login error (multiple users)",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    # P1, due 09:30 -> BREACHED -> HITL prompt: type 'n'
    {"number": "INC0001010", "description": "Exchange server high CPU alert - email delays",
     "category": "Server", "priority": "P1", "sla_due": "2024-01-15 09:30:00"},
    # P2, 90 min left of 240 (38%) -> AT_RISK -> work note only.
    # Step 5: change sla_due to '2024-01-15 10:00:00' -> BREACHED -> auto-escalated to L2-Network-Ops (no HITL for P2)
    {"number": "INC0001001", "description": "VPN not connecting after password change",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-10 12:00:00"},
    # P3, due in 2 days -> ON_TRACK -> monitored only
    {"number": "INC0001003", "description": "Laptop running very slowly",
     "category": "Hardware", "priority": "P3", "sla_due": "2024-01-17 09:00:00"},
]

if __name__ == "__main__":
    print(f"Model: {MODEL} | Simulated now: {NOW:%Y-%m-%d %H:%M}")
    outcomes = [monitor_ticket(t) for t in TEST_TICKETS]
    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    for o in outcomes:
        print(f"  {o['ticket_number']:<12} {o['breach_risk']:<9} {o['minutes_remaining']:>6} min  "
              f"escalated_to={o['escalated_to'] or '-':<16} hitl={o['hitl'] or '-'}")
    print(f"\nHITL decisions are logged to {AUDIT_LOG.relative_to(PROJECT_ROOT)}")
