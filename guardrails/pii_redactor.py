"""
ISDO Lab C9 - PII Redaction Middleware + Audit Trail
Masks PII before any ticket data is sent to Claude, and restores it for the system of record.

Detected (in this priority order - an earlier match wins where two overlap):
    EMAIL        john.smith@zensar.com
    USERNAME     ZENSAR\\jsmith01, "username: rwaghmare", "login id priya_k", "user r.waghmare", @jsmith
    EMPLOYEE_ID  ZEN-9823, EMP-00142
    IP_ADDRESS   192.168.1.45
    PHONE        +91-9876543210, 98765 43210, 555-123-4567
    NAME         spaCy NER (if en_core_web_sm is installed) + cue-word rules that work without spaCy
                 ("User John Smith", "for Michael D'Souza", "Mr Rao") + names derived from emails
Every value found once is masked everywhere it appears, and the same value always gets the same token.
Ticket references (INC0001001, REQ-1002, CHG...) are never masked.

Usage:
    from guardrails.pii_redactor import redact, restore
    clean_text, mapping = redact(raw_text)      # send clean_text to Claude
    original = restore(claude_response, mapping)

Install the spaCy model once for better name detection (optional - rules still work without it):
    python -m spacy download en_core_web_sm
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    nlp, SPACY_AVAILABLE = None, False
    print("\u26a0  spaCy model 'en_core_web_sm' not found - names use rule-based detection only.\n"
          "   For better name detection run:  python -m spacy download en_core_web_sm")

# -- PATTERNS -----------------------------------------------------------------------

TICKET_REF = re.compile(r"\b(?:INC|CHG|PRB|RITM|TASK)\d{5,}\b|\bREQ-?\d+\b", re.I)    # never masked

EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
DOMAIN_USER = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,14}\\[A-Za-z][A-Za-z0-9._\-]{0,30}[A-Za-z0-9]")
HANDLE = re.compile(r"(?<![\w.@])@[A-Za-z][A-Za-z0-9._\-]{1,30}[A-Za-z0-9]")
EMPLOYEE_ID = re.compile(r"\b(?:EMP|ZEN)[-\s]?\d{3,6}\b", re.I)
IP_ADDRESS = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)(?!\.\d)")
PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s\-]?)?(?:\d{10}|\d{5}[\s\-]\d{5}|\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4})(?!\d)")

_UNAME = r"([A-Za-z][A-Za-z0-9._\-]{0,30}[A-Za-z0-9])"
# Keywords that always introduce a username, e.g. "username rwaghmare", "login id: priya_k"
USERNAME_STRONG = re.compile(
    r"(?i)\b(?:user\s?name|user[\s_\-]?id|login[\s_\-]?(?:id|name)|logon[\s_\-]?name|sam[\s_]?account[\s_]?name"
    r"|account[\s_\-]?name|ad[\s_\-]?(?:account|user|login)|uid|upn)\b\s*(?:is\s+|of\s+)?[:=#\-]?\s*['\"]?" + _UNAME)
# Weaker keywords ("user", "account", "login"): only if followed by ':' / '=' or the token looks like a
# username (contains '.', '_' or a digit) - so "User reports VPN failure" is NOT treated as a username.
USERNAME_WEAK = re.compile(r"(?i)\b(?:user|account|login|logon)\b\s*(?:([:=#])\s*['\"]?" + _UNAME +
                           r"|['\"]?([A-Za-z][A-Za-z0-9]*[._\d][A-Za-z0-9._\-]*[A-Za-z0-9]))")

# Name rules (work without spaCy): a cue word followed by capitalised words.
NAME_WORD = r"[A-Z][a-z]*(?:['\u2019\-][A-Z]?[a-z]+)?"
NAME_CUE = re.compile(r"(?i:\b(user|contractor|employee|requester|requestor|caller|colleague|manager|engineer|"
                      r"for|by|from|contact|cc|attn|dear|hi|hello|regards|thanks|name\s*:|mr\.?|mrs\.?|ms\.?|dr\.?))"
                      r"\s+(" + NAME_WORD + r"(?:\s+" + NAME_WORD + r"){0,2})")
SINGLE_NAME_CUES = {"mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "dear", "hi", "hello", "name:", "name :"}

# Capitalised words that are NOT names (IT terms, teams, titles...) - stops "for Finance Team" being masked.
NOT_A_NAME = {w.lower() for w in """
    a an the new old all any some our your their my this that these those it we you they he she i
    user users team teams support service desk servicedesk it ops network finance hr sales marketing admin
    contractor contractors employee manager engineer requester caller colleague customer client vendor
    vpn sap erp crm ad mfa sso dns dhcp wifi wi-fi lan wan email outlook exchange teams zoom webex windows mac
    macos linux office microsoft cisco anyconnect salesforce sharepoint onedrive jira servicenow oracle sql
    laptop desktop printer server switch router password account access login logon reset request ticket
    issue error incident problem change update install upgrade monday tuesday wednesday thursday friday
    saturday sunday january february march april may june july august september october november december
    please urgent asap today tomorrow yesterday building floor room project phoenix board level director
    """.split()}
# Ordinary words that can follow a username keyword without being a username ("AD account after 5 attempts")
COMMON_WORDS = set("""
    is was were be been being has have had and or but not no cannot can't can could will would should may might
    must does did do the a an to in on at of with for from by as into onto after before since until during while
    when where because if then than again still also just only now here there this that these those it its
    locked unlocked disabled enabled expired reset missing required blocked deleted created needs need needed
    keeps keep gets got getting shows showing seems appears fails failed failing works working not-working
    password access details settings issue issues problem error errors please""".split())

# -- AUDIT (redaction events) -------------------------------------------------------

audit_log = []


def _audit(action, detail):
    entry = {"timestamp": datetime.now().isoformat(), "module": "PIIRedactor", "action": action, "detail": detail}
    audit_log.append(entry)
    return entry

# -- DETECTION -----------------------------------------------------------------------

def _is_ticket_ref(value):
    return bool(TICKET_REF.fullmatch(value))


def _name_ok(words):
    return all(w.lower().strip("'\u2019") not in NOT_A_NAME for w in words)


def _find_spans(text):
    """Return a list of (start, end, label) for every PII occurrence, non-overlapping."""
    spans = []

    def add(start, end, label):
        value = text[start:end]
        if end <= start or _is_ticket_ref(value):
            return
        if any(start < e and s < end for s, e, _ in spans):          # overlaps an earlier (higher-priority) span
            return
        spans.append((start, end, label))

    for m in EMAIL.finditer(text):
        add(m.start(), m.end(), "EMAIL")
    for m in DOMAIN_USER.finditer(text):
        add(m.start(), m.end(), "USERNAME")
    for m in USERNAME_STRONG.finditer(text):
        if m.group(1).lower() not in COMMON_WORDS:
            add(m.start(1), m.end(1), "USERNAME")
    for m in USERNAME_WEAK.finditer(text):
        grp = 2 if m.group(2) else 3
        if m.group(grp) and m.group(grp).lower() not in COMMON_WORDS:
            add(m.start(grp), m.end(grp), "USERNAME")
    for m in HANDLE.finditer(text):
        add(m.start(), m.end(), "USERNAME")
    for m in EMPLOYEE_ID.finditer(text):
        add(m.start(), m.end(), "EMPLOYEE_ID")
    for m in IP_ADDRESS.finditer(text):
        add(m.start(), m.end(), "IP_ADDRESS")
    for m in PHONE.finditer(text):
        add(m.start(), m.end(), "PHONE")

    # Names - spaCy NER first (if available), then cue-word rules
    if SPACY_AVAILABLE:
        for ent in nlp(text).ents:
            if ent.label_ == "PERSON" and not ent.text.isupper() and _name_ok(ent.text.split()):
                add(ent.start_char, ent.end_char, "NAME")
    for m in NAME_CUE.finditer(text):
        cue = m.group(1).lower()
        words = m.group(2).split()
        keep = []
        for w in words:                                # stop at the first word that isn't a name
            if not _name_ok([w]):
                break
            keep.append(w)
        if len(keep) >= 2 or (len(keep) == 1 and cue in SINGLE_NAME_CUES):
            start = m.start(2)
            end = start + len(" ".join(keep)) if len(keep) == len(words) else text.index(keep[-1], start) + len(keep[-1])
            add(start, end, "NAME")

    # Names derived from email addresses: john.smith@... also masks "John Smith" / "john smith" elsewhere
    for s, e, label in list(spans):
        if label == "EMAIL":
            parts = [p for p in re.split(r"[._\-]", text[s:e].split("@")[0]) if p.isalpha() and len(p) > 1]
            if len(parts) >= 2:
                for m in re.finditer(r"\b" + r"\s+".join(map(re.escape, parts)) + r"\b", text, re.I):
                    add(m.start(), m.end(), "NAME")

    # Mask every other occurrence of a username / name we already found
    for s, e, label in list(spans):
        if label in ("USERNAME", "NAME"):
            for m in re.finditer(r"(?<![\w.\\])" + re.escape(text[s:e]) + r"(?![\w])", text, re.I):
                add(m.start(), m.end(), label)

    return sorted(spans)


def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns (clean_text, mapping) where mapping restores the originals, e.g.
      redact("Contact john.doe@corp.com or call 9876543210")
      -> ("Contact [EMAIL_1] or call [PHONE_1]", {"[EMAIL_1]": "john.doe@corp.com", "[PHONE_1]": "9876543210"})
    The same value always gets the same token within one call.
    """
    if not text:
        return text, {}
    mapping, value_to_token, counters, out, pos = {}, {}, {}, [], 0
    for start, end, label in _find_spans(text):
        value = text[start:end]
        key = (label, value.lower())
        if key not in value_to_token:
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            value_to_token[key] = token
            mapping[token] = value
        out.append(text[pos:start] + value_to_token[key])
        pos = end
    clean = "".join(out) + text[pos:]
    _audit("redact", f"{len(mapping)} PII item(s) masked: {list(mapping.keys())}" if mapping else "No PII detected")
    return clean, mapping


def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for the system of record only - never send to Claude)."""
    restored = text
    for token in sorted(mapping, key=len, reverse=True):              # [NAME_10] before [NAME_1]
        restored = restored.replace(token, mapping[token])
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored


def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# -- AUDIT TRAIL LOGGER --------------------------------------------------------------

class AuditLogger:
    """Logs every agent action (timestamp, agent, tool, rationale, approval) to a JSONL file.
    The rationale is PII-redacted before it is written, so the audit trail never stores raw PII."""

    def __init__(self, log_file: str = str(PROJECT_ROOT / "logs" / "audit_trail.jsonl")):
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path                       # relative paths are from the project root
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = str(log_path)
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        safe_rationale = redact(rationale)[0][:200] if rationale else ""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": safe_rationale,
            "approval_status": approval_status,
        }
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'=' * 55}\nFULL AUDIT TRAIL ({len(self.entries)} entries)\n{'=' * 55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# -- DEMO ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print(f"PII REDACTION DEMO  (spaCy NER: {'ON' if SPACY_AVAILABLE else 'OFF - rule-based names'})")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
        # username cases
        "Username: rwaghmare cannot log in. AD account ZENSAR\\jsmith01 is locked.",
        "Please reset the password for user r.waghmare (login id: priya_k). Ping @asharma on Teams.",
        "User reports Outlook not syncing for the Finance Team since Monday.",      # no PII - must stay as is
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")
            assert restore(clean, mapping) == ticket, "restore() did not round-trip"

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 - VPN failure after password change for user r.waghmare", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK - 90 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to john.smith@zensar.com - auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: {logger.log_file}")
