"""
ISDO Lab C6/C7 - LangGraph Orchestrator (Supervisor) with extended HITL gate
Routes a ticket through the agents built in Labs C3-C5 as nodes of one StateGraph:

    triage -> resolution -> sla --(hitl_required)--> hitl -> communication -> END
                                 \\--(otherwise)-------------> communication -> END

HITL triggers (Lab C7) - any one is enough, all that apply are shown to the approver:
    P1_SLA          P1 ticket whose SLA is CRITICAL or BREACHED
    LOW_CONFIDENCE  Resolution Agent confidence is LOW (any priority)
    ACCESS_GRANT    request_type is 'Access Grant' (security-sensitive, any priority)

Run from the project root (start mcp_server/snow_shim.py and jira_shim.py first):
    python orchestrator/supervisor.py              # 4 test tickets: P2 VPN, P1 SAP, P3 Webex (LOW), access grant
    python orchestrator/supervisor.py --step3      # C7 Step 3 as written: VPN ticket rewritten as a Webex issue
"""
import contextlib
import csv
import io
import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))                               # so "agents" imports work
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")                        # never crash on a console glyph

from langgraph.graph import END, START, StateGraph

from agents import triage_agent                                     # Lab C3 - classify_ticket tool + prompt
from agents.resolution_agent import resolve_ticket                 # Lab C4 - ChromaDB RAG + confidence policy
from agents.sla_agent import SLA_TARGET_MIN, get_sla_status, update_ticket   # Lab C5 - SLA rules + ServiceNow PATCH

JIRA_URL = "http://localhost:5002/rest/api/2/issue"                 # Lab C2 Jira shim
REQUESTS_CSV = PROJECT_ROOT / "data" / "requests.csv"
PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
ESCALATION_TEAM = {"Network": "L2-Network-Ops", "Application": "L2-App-Support", "Server": "L2-Server-Ops",
                   "Access": "L2-Security-Ops"}   # anything else -> L2-Service-Desk

# ======================================================================
# STATE - one shared TypedDict. Nodes return ONLY the fields they own;
# audit_log uses a reducer (operator.add) so every node's entries are appended.
# ======================================================================
class TicketState(TypedDict, total=False):
    # input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str             # optional - looked up from Jira for REQ- tickets if missing
    # triage node
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    effective_priority: str
    # resolution node
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    confidence_score: float
    # sla node
    sla_breach_risk: str
    sla_minutes_remaining: int
    escalation_required: bool
    hitl_required: bool
    hitl_triggers: list           # machine-readable: ["P1_SLA", "LOW_CONFIDENCE", "ACCESS_GRANT"]
    hitl_reason: str              # human-readable, shown in the approval prompt (Lab C7)
    escalated_to: str
    # hitl node
    hitl_approved: bool
    access_granted: bool
    # communication node
    user_message: str
    final_status: str
    # every node
    audit_log: Annotated[list, operator.add]


def audit(agent, action, detail):
    print(f"  [AUDIT] {agent}: {action} - {detail}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def run_quietly(fn, *args):
    """Run a C4 agent, re-printing its console output indented under the node header."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args)
    for line in buf.getvalue().splitlines():
        if line.strip() and not set(line.strip()) <= {"="}:
            print(f"    | {line.strip()}")
    return result


def team_for(state):
    return ESCALATION_TEAM.get(state.get("triage_category") or state.get("category"), "L2-Service-Desk")


def lookup_request_type(ticket_number):
    """Request type for REQ- tickets: Jira shim first, then data/requests.csv."""
    if not ticket_number.startswith("REQ-"):
        return ""
    try:
        r = requests.get(f"{JIRA_URL}/{ticket_number}", timeout=3)
        if r.ok:
            return r.json()["fields"]["issuetype"]["name"] or ""
    except requests.RequestException:
        pass
    try:
        with open(REQUESTS_CSV, newline="", encoding="utf-8") as f:
            return next((row["request_type"] for row in csv.DictReader(f) if row["key"] == ticket_number), "")
    except FileNotFoundError:
        return ""


def approve_access_in_jira(ticket_number):
    """Mark an access request approved in the Jira shim (simulated if the shim isn't running)."""
    try:
        r = requests.put(f"{JIRA_URL}/{ticket_number}", timeout=3,
                         json={"fields": {"status": {"name": "Approved"}}})
        r.raise_for_status()
        source = "jira-shim"
    except requests.RequestException:
        source = "simulated (jira shim not running)"
    print(f"  [Jira Mock] APPROVED access request {ticket_number}  ({source})")

# ======================================================================
# NODE 1 - TRIAGE (Lab C3's classify_ticket tool - the classification comes back as the
# tool's schema-validated input, so there is no text parsing)
# ======================================================================
def classify(state, attempts=3):
    """Offer classify_ticket as the only tool. (Forcing it with tool_choice is rejected by
    current Opus models, so if Claude answers in prose we simply ask again.)"""
    classify_tool = next(t for t in triage_agent.tools if t["name"] == "classify_ticket")
    messages = [{"role": "user", "content": f"Please triage this ticket:\n\nTicket: {state['ticket_number']}\n"
                 f"Summary: {state['short_description']}\nDetails: {state['description']}"}]
    for _ in range(attempts):
        response = triage_agent.client.messages.create(
            model=triage_agent.MODEL, max_tokens=1024, output_config={"effort": "low"},
            system=triage_agent.SYSTEM_PROMPT, tools=[classify_tool], messages=messages)
        call = next((b for b in response.content if b.type == "tool_use"), None)
        if call is not None:
            return dict(call.input)
        messages += [{"role": "assistant", "content": response.content},
                     {"role": "user", "content": "Record the classification now by calling the classify_ticket tool."}]
    return None


def triage_node(state: TicketState) -> dict:
    print(f"\n▶ TRIAGE AGENT - {state['ticket_number']}")
    result = classify(state)
    if result is None:   # model gave no classification -> keep the ticket's own values, flag for review
        result = {"category": state["category"], "priority": state["priority"],
                  "assignment_group": "Service-Desk", "pii_detected": False, "reasoning": "triage failed - defaults used"}
    # Guardrail: triage may raise severity but never lower it below what the ticket already says,
    # otherwise a P1 mis-classified as P2 would silently skip the HITL gate.
    effective = min(result["priority"], state["priority"], key=PRIORITY_RANK.get)
    print(f"  Category: {result['category']}  Priority: {result['priority']}  (effective: {effective})")
    print(f"  Assign To: {result['assignment_group']}  PII: {result['pii_detected']}")
    return {"triage_category": result["category"], "triage_priority": result["priority"],
            "triage_assignment_group": result["assignment_group"], "pii_detected": result["pii_detected"],
            "effective_priority": effective,
            "audit_log": audit("TriageAgent", "classify_ticket",
                               f"{result['category']}/{result['priority']} -> {result['assignment_group']}"
                               f" | PII={result['pii_detected']} | {result['reasoning'][:60]}")}

# ======================================================================
# NODE 2 - RESOLUTION (Lab C4 agent: ChromaDB search + drafted steps)
# ======================================================================
def resolution_node(state: TicketState) -> dict:
    print("\n▶ RESOLUTION AGENT - searching KB")
    r = run_quietly(resolve_ticket, state["ticket_number"], state["short_description"], state["description"],
                    state["triage_category"], state["effective_priority"])
    print(f"  KB Article: {r['kb_article_used']}")
    print(f"  Confidence: {r['confidence']} ({r['top_score']:.0%}) | Auto-resolve: {r['auto_resolve']}")
    return {"kb_article": r["kb_article_used"], "resolution_text": r["resolution_text"],
            "auto_resolve": r["auto_resolve"], "confidence": r["confidence"], "confidence_score": r["top_score"],
            "audit_log": audit("ResolutionAgent", "search_kb",
                               f"{r['kb_article_used']} {r['confidence']} ({r['top_score']:.0%}) auto_resolve={r['auto_resolve']}")}

# ======================================================================
# NODE 3 - SLA + HITL TRIGGER EVALUATION (deterministic - no LLM)
# ======================================================================
def sla_node(state: TicketState) -> dict:
    print("\n▶ SLA AGENT - checking deadline and HITL triggers")
    priority = state["effective_priority"]
    sla = get_sla_status(state["ticket_number"], state["sla_due"], priority)
    escalate = sla["requires_escalation"]                      # CRITICAL/BREACHED and P1/P2
    request_type = state.get("request_type") or lookup_request_type(state["ticket_number"])

    triggers, reasons = [], []
    if escalate and priority == "P1":
        triggers.append("P1_SLA")
        reasons.append(f"P1 SLA {sla['breach_risk']} ({sla['minutes_remaining']} min) - escalation to {team_for(state)}")
    if state.get("confidence") == "LOW":
        triggers.append("LOW_CONFIDENCE")
        reasons.append(f"LOW KB confidence ({state.get('confidence_score', 0):.0%}) - no clear fix, route to L2")
    # Lab brief: category 'Access' + request_type 'Access Grant'. The request_type alone is what makes it
    # security-sensitive, so a grant that triage files under another category still cannot skip approval.
    if request_type == "Access Grant":
        triggers.append("ACCESS_GRANT")
        reasons.append(f"ACCESS GRANT - {state['short_description']} requires security approval")
    hitl = bool(triggers)
    hitl_reason = "; ".join(reasons)

    print(f"  SLA Risk: {sla['breach_risk']} | Minutes remaining: {sla['minutes_remaining']}"
          + (f" | Request type: {request_type}" if request_type else ""))
    print(f"  Escalation needed: {escalate} | HITL required: {hitl}" + (f" ({', '.join(triggers)})" if hitl else ""))
    out = {"sla_breach_risk": sla["breach_risk"], "sla_minutes_remaining": sla["minutes_remaining"],
           "escalation_required": escalate, "hitl_required": hitl, "hitl_triggers": triggers,
           "hitl_reason": hitl_reason, "request_type": request_type,
           "audit_log": audit("SLAAgent", "get_sla_status",
                              f"{sla['breach_risk']} ({sla['minutes_remaining']} min) escalate={escalate}"
                              f" hitl={hitl} triggers={triggers or '-'}")}
    if escalate and priority != "P1":                          # P2 breach: escalate straight away (C5 policy)
        update_ticket(state["ticket_number"], "escalate", team_for(state), f"Auto-escalated: SLA {sla['breach_risk']}")
        out["escalated_to"] = team_for(state)
        out["audit_log"] += audit("SLAAgent", "update_ticket", f"escalated to {team_for(state)} (no HITL needed for {priority})")
    return out

# ======================================================================
# NODE 4 - HITL GATE (one human decision covering every trigger that fired)
# ======================================================================
def hitl_node(state: TicketState) -> dict:
    triggers, team = state["hitl_triggers"], team_for(state)
    actions = []
    if ("P1_SLA" in triggers or "LOW_CONFIDENCE" in triggers) and not state.get("escalated_to"):
        actions.append(f"escalate to {team}")
    if "ACCESS_GRANT" in triggers:
        actions.append("grant the requested access")
    print("\n▶ HITL GATE - human approval required")
    print(f"  {'WARNING ' * 8}\n  Ticket:  {state['ticket_number']} | Priority: {state['effective_priority']}")
    for i, reason in enumerate(state["hitl_reason"].split("; "), 1):
        print(f"  Reason{f' {i}' if len(triggers) > 1 else ''}: {reason}")
    print(f"  KB:      {state['kb_article']} ({state['confidence']})")
    print(f"  If approved: {', '.join(actions) or 'no further action'}\n  {'WARNING ' * 8}")
    try:
        approved = input("  Approve action? [y/n]: ").strip().lower() == "y"
    except EOFError:                                           # no human available -> never auto-approve
        approved = False
    decision = "APPROVED" if approved else "REJECTED"
    print(f"  Decision: {decision}")
    out = {"hitl_approved": approved, "access_granted": approved and "ACCESS_GRANT" in triggers,
           "audit_log": audit("HITLGate", "approval_decision",
                              f"{decision} by human operator | triggers={','.join(triggers)} | {state['hitl_reason']}")}
    if approved:                                               # the approved actions actually happen here
        if f"escalate to {team}" in actions:
            update_ticket(state["ticket_number"], "escalate", team, f"Approved by human operator: {state['hitl_reason']}")
            out["escalated_to"] = team
            out["audit_log"] += audit("HITLGate", "update_ticket", f"escalated to {team}")
        if "ACCESS_GRANT" in triggers:
            approve_access_in_jira(state["ticket_number"])
            out["audit_log"] += audit("HITLGate", "approve_access", f"access request {state['ticket_number']} approved")
    return out

# ======================================================================
# NODE 5 - COMMUNICATION (message to the requester + final status)
# ======================================================================
def communication_node(state: TicketState) -> dict:
    print("\n▶ COMMUNICATION AGENT - drafting user message")
    n, triggers = state["ticket_number"], state.get("hitl_triggers", [])
    if state.get("hitl_required") and not state.get("hitl_approved"):
        status = "AWAITING_REVIEW"
        if "ACCESS_GRANT" in triggers:
            msg = (f"Dear Requester,\n\nYour access request {n} is pending security approval. No access has "
                   f"been granted yet - we will contact you once it has been reviewed.\n\nIT Support Team")
        else:
            msg = (f"Dear User,\n\nYour ticket {n} is pending approval from our senior support team, who will "
                   f"contact you shortly. Reference: {n}\n\nIT Support Team")
    elif state.get("access_granted"):
        status = "APPROVED"
        msg = (f"Dear Requester,\n\nYour access grant request {n} has been approved by the security team. "
               f"Access will be provisioned by {state['triage_assignment_group']} and you will receive "
               f"connection details separately.\n\nIT Support Team")
    elif state.get("escalated_to"):
        status = "ESCALATED"
        why = ("our KB has no confirmed fix for this issue, so a specialist will investigate"
               if "LOW_CONFIDENCE" in triggers else f"it is a priority ({state['effective_priority']}) incident")
        msg = (f"Dear User,\n\nYour ticket {n} has been escalated to {state['escalated_to']} because {why}. "
               f"An engineer is working on it now.\n\nIT Support Team")
    elif state.get("auto_resolve") and not state.get("hitl_required"):   # self-service only if no HITL trigger
        status = "RESOLVED"
        msg = (f"Dear User,\n\nRegarding your ticket {n}, please follow these steps:\n\n"
               f"{state['resolution_text']}\n\nIf the issue persists, just reply and we will reopen it.\n\nIT Support Team")
        update_ticket(n, "update_state", new_state="Resolved")
    else:
        status = "ASSIGNED"
        hours = SLA_TARGET_MIN.get(state["effective_priority"], 480) // 60
        msg = (f"Dear User,\n\nYour ticket {n} has been assigned to {state['triage_assignment_group']}. "
               f"Expected resolution: within {hours} hours.\n\nIT Support Team")
    print(f"\n  USER MESSAGE:\n  {'-' * 50}")
    for line in msg.splitlines():
        print(f"  {line}")
    print(f"  {'-' * 50}")
    return {"user_message": msg, "final_status": status,
            "audit_log": audit("CommunicationAgent", "post_comment", f"final_status={status} ({len(msg)} chars)")}

# ======================================================================
# ROUTING + GRAPH
# ======================================================================
def route_after_sla(state: TicketState) -> Literal["hitl", "communication"]:
    return "hitl" if state.get("hitl_required") else "communication"


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)
    g.add_edge(START, "triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)
    return g.compile()


GRAPH = build_graph()

# ======================================================================
# RUN - simulated now is 2024-01-15 10:30 (same clock as Lab C5)
# ======================================================================
TEST_TICKETS = [
    # P2 VPN: 90 min left of 240 -> AT_RISK, KB match HIGH -> auto-resolve, no HITL
    {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
     "description": "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
    # P1 SAP outage: 10 min left of 60 -> CRITICAL -> HITL (P1_SLA). Run once with 'y', once with 'n' (Step 2)
    {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
     "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    # Step 3 - no KB coverage -> LOW confidence -> HITL (LOW_CONFIDENCE) even though it is only P3
    {"ticket_number": "TEST-004", "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
     "description": "Cisco Webex not launching on MacBook M2 after Sonoma update.",
     "category": "Software", "priority": "P3", "sla_due": "2024-01-17 09:00:00"},
    # Step 4 - access grant: request_type is looked up from the Jira shim (REQ-1002 = 'Access Grant')
    {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
     "description": "Contractor needs VPN access. Email: contractor@client.com",
     "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00"},
]

# Step 3 - the VPN ticket rewritten as an issue the KB does not cover. BOTH fields change: the KB search
# uses summary + details, so a description that still mentions VPN would keep the match HIGH.
STEP3_TICKET = {**TEST_TICKETS[0],
                "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
                "description": "Cisco Webex app crashes on launch on a MacBook M2 since the macOS Sonoma update."}

if __name__ == "__main__":
    tickets = [STEP3_TICKET] if "--step3" in sys.argv else TEST_TICKETS
    results = []
    for ticket in tickets:
        print(f"\n{'=' * 60}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'=' * 60}")
        final = GRAPH.invoke({**ticket, "audit_log": []})
        print(f"\nFINAL STATUS: {final['final_status']}")
        results.append(final)

    print(f"\n{'=' * 60}\nAUDIT LOG (Step 5)\n{'=' * 60}")
    for final in results:
        print(f"\n{final['ticket_number']}  ->  {final['final_status']}"
              + (f"   [HITL: {'APPROVED' if final.get('hitl_approved') else 'REJECTED'}]" if final.get("hitl_required") else ""))
        for e in final["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<18} {e['action']:<18} {e['detail']}")
