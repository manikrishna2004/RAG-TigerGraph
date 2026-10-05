import json, os, re, unicodedata
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_MON = "|".join(MONTHS)

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’`]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.json")
try:
    EVENTS = json.load(open(_PATH, encoding="utf-8"))["events"]
except Exception as _err:                       # a missing file must not crash the agent at import
    print(f"  [events] events.json unavailable ({_err}); event lookup disabled")
    EVENTS = []
PREV = {}
for e in EVENTS:
    e["year"] = int(e["year"])
    e["_event"] = norm(e["event"])
    e["_venue"] = norm(e["venue"])
    e["_sport"] = norm(e["sport"])
    if e.get("prev"):
        PREV[(e["season"].lower(), e["year"])] = int(e["prev"])

def _has(qn, phrase):
    """Whole-word phrase match, so 'sydney' doesn't match inside a longer venue name."""
    return bool(phrase) and f" {phrase} " in f" {qn} "

def _parse_date(q):
    ql = q.lower()
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\b(?:,?\s+(\d{{4}}))?", ql)
    if m:
        return int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)) if m.group(3) else None
    m = re.search(rf"\b({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", ql)
    if m:
        return int(m.group(2)), MONTHS[m.group(1)], int(m.group(3)) if m.group(3) else None
    return None

def _date_match(e, d):
    """5 if the event is a single day on that date, 3 if the date falls inside a longer range."""
    try:
        s = date.fromisoformat(e["start"]); t = date.fromisoformat(e["end"])
        day, month, year = d
        target = date(year or s.year, month, day)
    except Exception:
        return 0
    if not (s <= target <= t):
        return 0
    return 5 if s == t else 3

def lookup_events(question, top_k=6):
    ql = question.lower()
    qn = norm(question)
    years = {int(y) for y in re.findall(r"\b(?:18|19|20)\d\d\b", question)}
    season = "winter" if "winter" in qn else "summer" if "summer" in qn else None

    # "...Olympics held immediately before 2016" -> look at the previous edition instead
    m = re.search(r"immediately\s+(?:before|preceding|prior\s+to)\s+((?:18|19|20)\d\d)", ql)
    if m:
        x = int(m.group(1))
        prev = PREV.get((season, x)) or next((p for (s, y), p in PREV.items() if y == x), None)
        years.discard(x)
        if prev:
            years.add(prev)

    d = _parse_date(question)
    scored = []
    for e in EVENTS:
        s, sig = 0, set()
        if e["year"] in years: s += 3; sig.add("year")
        if _has(qn, e["_venue"]): s += 2 + min(len(e["_venue"].split()), 4); sig.add("venue")
        if d and _date_match(e, d): s += _date_match(e, d); sig.add("date")
        if _has(qn, e["_event"]): s += 1 + min(len(e["_event"].split()), 3); sig.add("event")
        if _has(qn, e["_sport"]): s += 2; sig.add("sport")
        if season and e["season"].lower() == season: s += 1
        if len(sig) >= 2:          # one weak signal (e.g. only the sport) is not a match
            scored.append((s, e))
    if d:                                  # a date in the question is a constraint, not just a hint
        dated = [(s, e) for s, e in scored if _date_match(e, d)]
        if dated:
            scored = dated
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    best = scored[0][0]
    return [e for s, e in scored[:top_k] if s >= best - 1]

def as_fact(e):
    fact = (
        f"{e['title']}: venue {e['venue']}; held {e['start']} to {e['end']}; "
        f"gold: {e['gold']} ({e['gold_noc']}); silver: {e['silver']} ({e['silver_noc']}); "
        f"bronze: {e['bronze']} ({e['bronze_noc']})")
    extra = ", ".join(f"{k}: {e[k]}" for k in ("competitors", "nations") if e.get(k) not in (None, "", 0))
    return {"fact_str": fact + (f"; {extra}" if extra else "")}


_METRICS = ("competitors", "nations")


def aggregate_events(spec):
    """Generic filter + aggregate over the per-event table (no question templates).
    spec keys (all optional): sport, year, season, gender ('men'|'women'), event_text, metric
    ('competitors'|'nations'), op ('count_over'|'max'|'min'|'count'|'sum'|'list'), threshold."""
    if not EVENTS or not isinstance(spec, dict):
        return None
    sel = EVENTS
    try:
        if spec.get("year"):
            sel = [e for e in sel if e["year"] == int(spec["year"])]
        thr = float(spec["threshold"]) if spec.get("threshold") not in (None, "") else 0.0
    except (TypeError, ValueError):
        return None
    if spec.get("season"):
        sel = [e for e in sel if e["season"].lower() == str(spec["season"]).lower()]
    if spec.get("sport"):
        sp = norm(spec["sport"])
        sel = [e for e in sel if sp and (e["_sport"] == sp or sp in e["_sport"] or e["_sport"] in sp)]
    if spec.get("gender") in ("men", "women"):
        tok = "mens" if spec["gender"] == "men" else "womens"
        sel = [e for e in sel if tok in e["_event"].split()]
    if spec.get("event_text"):
        need = set(norm(spec["event_text"]).split())
        sel = [e for e in sel if need <= set(e["_event"].split())]

    metric = spec.get("metric") if spec.get("metric") in _METRICS else None
    op = str(spec.get("op") or "").lower()
    desc = ", ".join(f"{k}={spec[k]}" for k in ("sport", "year", "season", "gender", "event_text") if spec.get(k)) or "no filter"
    if not sel:
        return f"[aggregate] No event pages match the filter ({desc})."

    lines = [f"[aggregate] filter: {desc}; {len(sel)} event pages match."]
    vals = [(e, e[metric]) for e in sel if metric and e.get(metric) is not None]
    if op == "count_over" and metric:
        over = sorted([(e, v) for e, v in vals if v > thr], key=lambda x: -x[1])
        lines.append(f"{len(over)} events have more than {thr:g} {metric}:")
        lines += [f"  - {e['title']}: {v}" for e, v in over[:40]]
    elif op in ("max", "min") and vals:
        best = max(v for _, v in vals) if op == "max" else min(v for _, v in vals)
        top = [(e, v) for e, v in vals if v == best]
        lines.append(f"{op} {metric} = {best:g} for: " + "; ".join(e["title"] for e, _ in top[:10]))
    elif op == "sum" and vals:
        lines.append(f"total {metric} across {len(vals)} events = {sum(v for _, v in vals):g}")
    elif op == "list" and vals:
        lines += [f"  - {e['title']}: {metric}={v}" for e, v in vals[:40]]
    else:                                   # 'count' or an unrecognised op: report the match count
        lines.append(f"count = {len(sel)}")
        lines += [f"  - {e['title']}" for e in sel[:25]]
    if metric:
        missing = len(sel) - len(vals)
        if missing:
            lines.append(f"({missing} matching pages have no {metric} value and are not counted)")
    return "\n".join(lines)