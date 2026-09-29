"""
ISDO Lab C6 - LangGraph Orchestrator (Supervisor)
Routes a ticket through the agents built in Labs C3-C5 as nodes of one StateGraph:

    triage -> resolution -> sla --(hitl_required)--> hitl -> communication -> END
                                 \\--(otherwise)-------------> communication -> END

Run from the project root (start mcp_server/snow_shim.py first so updates hit ServiceNow):
    python orchestrator/supervisor.py
"""
import contextlib
import io
import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "agents" imports work
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")                        # never crash on a console glyph

from langgraph.graph import END, START, StateGraph

from agents.triage_agent import triage_ticket                      # Lab C3 - tool-calling classifier
from agents.resolution_agent import resolve_ticket                 # Lab C4 - ChromaDB RAG + confidence policy
from agents.sla_agent import SLA_TARGET_MIN, get_sla_status, update_ticket   # Lab C5 - SLA rules + ServiceNow PATCH

PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

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
    escalated_to: str
    # hitl node
    hitl_approved: bool
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
    """Run a C3/C4 agent, re-printing its console output indented under the node header."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args)
    for line in buf.getvalue().splitlines():
        if line.strip() and not set(line.strip()) <= {"="}:
            print(f"    | {line.strip()}")
    return result


ESCALATION_TEAM = {"Network": "L2-Network-Ops", "Application": "L2-App-Support", "Server": "L2-Server-Ops",
                   "Access": "L2-Security-Ops"}   # anything else -> L2-Service-Desk

# ======================================================================
# NODE 1 - TRIAGE (Lab C3 agent: Claude tool call returns structured JSON)
# ======================================================================
def triage_node(state: TicketState) -> dict:
    print(f"\n▶ TRIAGE AGENT - {state['ticket_number']}")
    result = run_quietly(triage_ticket, state["ticket_number"], state["short_description"], state["description"])
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
# NODE 3 - SLA (Lab C5 rules; deterministic - no LLM needed to do date maths)
# ======================================================================
def sla_node(state: TicketState) -> dict:
    print("\n▶ SLA AGENT - checking deadline")
    priority = state["effective_priority"]
    sla = get_sla_status(state["ticket_number"], state["sla_due"], priority)
    escalate = sla["requires_escalation"]                      # CRITICAL/BREACHED and P1/P2
    hitl = escalate and priority == "P1"                       # P1 escalations need a human
    team = ESCALATION_TEAM.get(state["triage_category"], "L2-Service-Desk")
    print(f"  SLA Risk: {sla['breach_risk']} | Minutes remaining: {sla['minutes_remaining']}")
    print(f"  Escalation needed: {escalate} | HITL required: {hitl}")
    out = {"sla_breach_risk": sla["breach_risk"], "sla_minutes_remaining": sla["minutes_remaining"],
           "escalation_required": escalate, "hitl_required": hitl,
           "audit_log": audit("SLAAgent", "get_sla_status",
                              f"{sla['breach_risk']} ({sla['minutes_remaining']} min) escalate={escalate} hitl={hitl}")}
    if escalate and not hitl:                                  # P2 breach: escalate straight away (C5 policy)
        update_ticket(state["ticket_number"], "escalate", team, f"Auto-escalated: SLA {sla['breach_risk']}")
        out["escalated_to"] = team
        out["audit_log"] += audit("SLAAgent", "update_ticket", f"escalated to {team} (no HITL needed for {priority})")
    return out

# ======================================================================
# NODE 4 - HITL GATE (human approval before any P1 escalation)
# ======================================================================
def hitl_node(state: TicketState) -> dict:
    team = ESCALATION_TEAM.get(state["triage_category"], "L2-Service-Desk")
    print("\n▶ HITL GATE - human approval required")
    print(f"  {'!!! ' * 8}\n  Ticket:  {state['ticket_number']} | Priority: {state['effective_priority']}")
    print(f"  Reason:  SLA {state['sla_breach_risk']} ({state['sla_minutes_remaining']} min) - escalate to {team}")
    print(f"  KB:      {state['kb_article']} ({state['confidence']})\n  {'!!! ' * 8}")
    try:
        approved = input("  Approve escalation? [y/n]: ").strip().lower() == "y"
    except EOFError:                                           # no human available -> never auto-approve
        approved = False
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    out = {"hitl_approved": approved,
           "audit_log": audit("HITLGate", "approval_decision",
                              f"{'APPROVED' if approved else 'REJECTED'} by human operator (escalate to {team})")}
    if approved:                                               # the approved action actually happens here
        update_ticket(state["ticket_number"], "escalate", team, "P1 escalation approved by human operator")
        out["escalated_to"] = team
        out["audit_log"] += audit("HITLGate", "update_ticket", f"escalated to {team}")
    return out

# ======================================================================
# NODE 5 - COMMUNICATION (message to the requester + final status)
# ======================================================================
def communication_node(state: TicketState) -> dict:
    print("\n▶ COMMUNICATION AGENT - drafting user message")
    n = state["ticket_number"]
    if state.get("hitl_required") and not state.get("hitl_approved"):
        status = "AWAITING_REVIEW"
        msg = (f"Dear User,\n\nYour ticket {n} is being reviewed by our senior support team, who will "
               f"contact you shortly. Reference: {n}\n\nIT Support Team")
    elif state.get("escalated_to"):                            # escalation beats self-service on a breached SLA
        status = "ESCALATED"
        msg = (f"Dear User,\n\nYour ticket {n} has been escalated to {state['escalated_to']} as a priority "
               f"({state['effective_priority']}). An engineer is working on it now.\n\nIT Support Team")
    elif state.get("auto_resolve"):
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
    # P1 SAP outage: 10 min left of 60 -> CRITICAL -> HITL gate (type 'y')
    {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
     "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
]

if __name__ == "__main__":
    results = []
    for ticket in TEST_TICKETS:
        print(f"\n{'=' * 60}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'=' * 60}")
        final = GRAPH.invoke({**ticket, "audit_log": []})
        print(f"\nFINAL STATUS: {final['final_status']}")
        results.append(final)

    print(f"\n{'=' * 60}\nAUDIT LOG (Step 5)\n{'=' * 60}")
    for final in results:
        print(f"\n{final['ticket_number']}  ->  {final['final_status']}")
        for e in final["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<18} {e['action']:<18} {e['detail']}")
