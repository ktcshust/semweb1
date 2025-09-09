#!/usr/bin/env python3
"""
graph_rag_tfl_full.py

Single-file pipeline:
- Ingest TfL data into Neo4j with 14 node types + relationships.
- Provide an interactive "ask" mode: natural language -> GPT-4 -> Cypher -> run on Neo4j -> show results.
- Introspection + validation + fallback heuristics to handle missing relationship types and other schema mismatches.

Requirements:
  pip install requests python-dotenv neo4j openai

.env must contain:
  TFL_APP_KEY="..."
  NEO4J_URI=bolt://127.0.0.1:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=...
  OPENAI_API_KEY=sk-...

Usage:
  python graph_rag_tfl_full.py --ingest
  python graph_rag_tfl_full.py --ask
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from openai import OpenAI

load_dotenv()

# --- Config / env ---
TFL_BASE = "https://api.tfl.gov.uk"
TFL_APP_KEY = os.getenv("TFL_APP_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. --ask will fail until set.", file=sys.stderr)

# OpenAI client (new API)
client = OpenAI(api_key=OPENAI_API_KEY)

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))

# --- TfL helper ---
def tfl_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    url = f"{TFL_BASE}{path}"
    p = dict(params or {})
    if TFL_APP_KEY:
        p["app_key"] = TFL_APP_KEY
    r = requests.get(url, params=p, timeout=timeout)
    r.raise_for_status()
    return r.json()

# --- Ensure constraints / indexes ---
def ensure_constraints_and_indexes():
    with driver.session() as s:
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Station) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Line) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:StopPoint) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Platform) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:FareZone) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Operator) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:BikePoint) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Incident) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Interchange) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Timetable) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:AccessibilityFeature) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Facility) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Landmark) REQUIRE n.id IS UNIQUE",
        ]
        for st in stmts:
            try:
                s.run(st)
            except Exception as e:
                # ignore failure in older neo4j versions
                print("Constraint warning:", e)

# --- Upsert helpers (14 node types + rels) ---
def upsert_line(tx, line_obj):
    tx.run("""
    MERGE (l:Line {id:$id})
    SET l.name=$name, l.mode=$mode, l.raw=$raw
    """, id=line_obj.get("id"), name=line_obj.get("name"), mode=line_obj.get("modeName") or line_obj.get("mode"), raw=json.dumps(line_obj, default=str))

def upsert_stop_point(tx, sp):
    spid = sp.get("id") or sp.get("naptanId")
    lat = sp.get("lat") or sp.get("latitude") or (sp.get("location") or {}).get("lat")
    lon = sp.get("lon") or sp.get("longitude") or (sp.get("location") or {}).get("lon")
    tx.run("""
    MERGE (sp:StopPoint {id:$id})
    SET sp.name=$name, sp.commonName=$commonName, sp.lat=$lat, sp.lon=$lon, sp.modes=$modes, sp.raw=$raw
    """, id=spid, name=sp.get("name"), commonName=sp.get("commonName"), lat=lat, lon=lon, modes=sp.get("modes"), raw=json.dumps(sp, default=str))

def upsert_station_from_stoppoint(tx, sp):
    spid = sp.get("id")
    station_id = sp.get("stationNaptan") or sp.get("parentId") or spid
    name = sp.get("stationName") or sp.get("commonName") or sp.get("name")
    lat = sp.get("lat") or sp.get("latitude") or (sp.get("location") or {}).get("lat")
    lon = sp.get("lon") or sp.get("longitude") or (sp.get("location") or {}).get("lon")
    tx.run("""
    MERGE (s:Station {id:$sid})
    SET s.name=$name, s.lat=$lat, s.lon=$lon, s.raw = coalesce(s.raw, $raw)
    """, sid=station_id, name=name, lat=lat, lon=lon, raw=json.dumps(sp, default=str))
    tx.run("""
    MATCH (sp:StopPoint {id:$spid}), (s:Station {id:$sid})
    MERGE (sp)-[:PART_OF]->(s)
    """, spid=spid, sid=station_id)

def upsert_platform(tx, child, station_id):
    pid = child.get("id") or child.get("naptanId")
    if not pid:
        return
    tx.run("""
    MERGE (p:Platform {id:$id})
    SET p.name=$name, p.raw=$raw
    WITH p
    MATCH (s:Station {id:$sid})
    MERGE (s)-[:HAS_PLATFORM]->(p)
    """, id=pid, name=child.get("commonName") or child.get("name"), raw=json.dumps(child, default=str), sid=station_id)

def upsert_farezone(tx, zone_name, station_id=None):
    if zone_name is None:
        return
    tx.run("MERGE (z:FareZone {name:$name})", name=str(zone_name))
    if station_id:
        tx.run("""
        MATCH (z:FareZone {name:$name}), (s:Station {id:$sid})
        MERGE (s)-[:IN_ZONE]->(z)
        """, name=str(zone_name), sid=station_id)

def upsert_operator(tx, op_name):
    if not op_name:
        return
    tx.run("MERGE (o:Operator {name:$name}) SET o.raw = coalesce(o.raw,$raw)", name=op_name, raw=op_name)

def upsert_accessibility_and_facilities(tx, sp):
    sid = sp.get("id") or sp.get("stationNaptan") or sp.get("parentId")
    for ap in sp.get("additionalProperties", []):
        key = (ap.get("key") or "").strip()
        val = (ap.get("value") or "").strip()
        if not key:
            continue
        name = f"{key}: {val}"
        if any(k in key.lower() for k in ['stepfree','accessible','lift','escalator']):
            tx.run("""
            MERGE (a:AccessibilityFeature {name:$name})
            WITH a
            MATCH (s:Station {id:$sid})
            MERGE (s)-[:HAS_ACCESSIBILITY]->(a)
            """, name=name, sid=sid)
        else:
            tx.run("""
            MERGE (f:Facility {name:$name})
            WITH f
            MATCH (s:Station {id:$sid})
            MERGE (s)-[:HAS_FACILITY]->(f)
            """, name=name, sid=sid)

def create_on_line_and_next_to(tx, line_id, stoppoint_list):
    prev_station_id = None
    for sp in stoppoint_list:
        station_id = sp.get("stationNaptan") or sp.get("id")
        if not station_id:
            continue
        tx.run("""
        MATCH (s:Station {id:$sid}), (l:Line {id:$lid})
        MERGE (s)-[:ON_LINE]->(l)
        """, sid=station_id, lid=line_id)
        if prev_station_id and station_id and prev_station_id != station_id:
            tx.run("""
            MATCH (s1:Station {id:$s1}), (s2:Station {id:$s2})
            MERGE (s1)-[:NEXT_TO {line:$line}]->(s2)
            MERGE (s2)-[:NEXT_TO {line:$line}]->(s1)
            """, s1=prev_station_id, s2=station_id, line=line_id)
        prev_station_id = station_id

def upsert_incident_from_line_status(tx, line_id, status_obj):
    eid = status_obj.get("id") or f"incident-{line_id}-{int(time.time()*1000)}"
    tx.run("""
    MERGE (inc:Incident {id:$id})
    SET inc.statusSeverity = $sev, inc.description = $desc, inc.raw = $raw
    WITH inc
    MATCH (l:Line {id:$line})
    MERGE (l)-[:HAS_INCIDENT]->(inc)
    """, id=eid, sev=status_obj.get("statusSeverity"), desc=status_obj.get("statusSeverityDescription") or status_obj.get("description") or status_obj.get("reason"), raw=json.dumps(status_obj, default=str), line=line_id)

def upsert_bikepoint(tx, bp):
    bid = bp.get("id") or bp.get("commonName")
    lat = bp.get("lat") or bp.get("latitude")
    lon = bp.get("lon") or bp.get("longitude")
    tx.run("""
    MERGE (b:BikePoint {id:$id})
    SET b.name=$name, b.lat=$lat, b.lon=$lon, b.raw=$raw
    """, id=bid, name=bp.get("commonName") or bp.get("name"), lat=lat, lon=lon, raw=json.dumps(bp, default=str))

def upsert_place_as_landmark(tx, place_obj, station_id=None):
    pid = place_obj.get("id") or place_obj.get("placeId") or place_obj.get("commonName")
    lat = place_obj.get("lat") or place_obj.get("latitude")
    lon = place_obj.get("lon") or place_obj.get("longitude")
    tx.run("""
    MERGE (lm:Landmark {id:$id})
    SET lm.name=$name, lm.category=$cat, lm.lat=$lat, lm.lon=$lon, lm.raw=$raw
    """, id=str(pid), name=place_obj.get("commonName") or place_obj.get("name"), cat=place_obj.get("placeType") or place_obj.get("category") or "place", lat=lat, lon=lon, raw=json.dumps(place_obj, default=str))
    if station_id:
        tx.run("""
        MATCH (s:Station {id:$sid}), (lm:Landmark {id:$id})
        MERGE (s)-[:NEAR]->(lm)
        """, sid=station_id, id=str(pid))

def upsert_interchange_by_detection(tx, station_id, lines_serving):
    if not lines_serving or len(lines_serving) < 2:
        return
    iid = f"interchange-{station_id}"
    tx.run("MERGE (i:Interchange {id:$id}) SET i.stationId=$sid, i.lines=$lines", id=iid, sid=station_id, lines=lines_serving)
    tx.run("MATCH (s:Station {id:$sid}), (i:Interchange {id:$id}) MERGE (s)-[:HAS_INTERCHANGE]->(i)", sid=station_id, id=iid)
    for ln in lines_serving:
        tx.run("MATCH (i:Interchange {id:$id}), (l:Line {id:$lid}) MERGE (i)-[:CONNECTS_LINE]->(l)", id=iid, lid=ln)

# --- Ingest orchestration ---
def ingest_all(radius_for_places: int = 200):
    ensure_constraints_and_indexes()
    try:
        lines = tfl_get("/Line/Mode/tube")
    except Exception as e:
        print("Failed to fetch lines:", e, file=sys.stderr)
        lines = []

    with driver.session() as s:
        for line in lines:
            lid = line.get("id")
            try:
                print("Upserting Line:", lid)
                s.write_transaction(upsert_line, line)
                try:
                    sp_list = tfl_get(f"/Line/{lid}/StopPoints")
                except Exception as e:
                    print(f"Failed StopPoints for {lid}:", e)
                    sp_list = []
                # upsert stopPoints & derived station/platform/facility/accessibility
                for sp in sp_list:
                    s.write_transaction(upsert_stop_point, sp)
                    s.write_transaction(upsert_station_from_stoppoint, sp)
                    for child in sp.get("children", []):
                        s.write_transaction(upsert_platform, child, sp.get("stationNaptan") or sp.get("id"))
                    # fare zones
                    if sp.get("fareZone"):
                        if isinstance(sp["fareZone"], list):
                            for z in sp["fareZone"]:
                                s.write_transaction(upsert_farezone, z, sp.get("stationNaptan") or sp.get("id"))
                        else:
                            s.write_transaction(upsert_farezone, sp["fareZone"], sp.get("stationNaptan") or sp.get("id"))
                    if sp.get("zone"):
                        s.write_transaction(upsert_farezone, sp.get("zone"), sp.get("stationNaptan") or sp.get("id"))
                    # accessibility & facilities
                    s.write_transaction(upsert_accessibility_and_facilities, sp)
                # create relations ON_LINE and NEXT_TO
                s.write_transaction(create_on_line_and_next_to, lid, sp_list)
                # line statuses -> incidents
                try:
                    statuses = tfl_get(f"/Line/{lid}/Status")
                    if isinstance(statuses, dict) and statuses.get("lineStatuses"):
                        for ls in statuses["lineStatuses"]:
                            s.write_transaction(upsert_incident_from_line_status, lid, ls)
                    elif isinstance(statuses, list):
                        for st in statuses:
                            if isinstance(st, dict) and st.get("lineStatuses"):
                                for ls in st["lineStatuses"]:
                                    s.write_transaction(upsert_incident_from_line_status, lid, ls)
                except Exception:
                    pass
                # places near first stop
                if sp_list:
                    first = sp_list[0]
                    lat = first.get("lat") or (first.get("location") or {}).get("lat")
                    lon = first.get("lon") or (first.get("location") or {}).get("lon")
                    if lat and lon:
                        try:
                            places = tfl_get("/Place", params={"lat": lat, "lon": lon, "radius": radius_for_places})
                            for p in places:
                                s.write_transaction(upsert_place_as_landmark, p, first.get("stationNaptan") or first.get("id"))
                        except Exception:
                            pass
            except Exception:
                print("Error on line:", lid)
                traceback.print_exc()

        # ingest bikepoints
        try:
            bikepoints = tfl_get("/BikePoint")
            for bp in bikepoints:
                s.write_transaction(upsert_bikepoint, bp)
        except Exception:
            pass

        # create detected interchanges where station serves >=2 lines
        try:
            s.run("""
            MATCH (s:Station)-[:ON_LINE]->(l:Line)
            WITH s, collect(DISTINCT l.id) AS lines
            WHERE size(lines) > 1
            MERGE (i:Interchange {id: 'interchange-' + s.id})
            SET i.stationId = s.id, i.lines = lines
            MERGE (s)-[:HAS_INTERCHANGE]->(i)
            WITH i, lines
            UNWIND lines AS lid
            MATCH (l:Line {id: lid})
            MERGE (i)-[:CONNECTS_LINE]->(l)
            """)
        except Exception:
            pass

        # create default operator link if missing (quick fix)
        try:
            s.run("""
            MERGE (op:Operator {name:'London Underground'})
            WITH op
            MATCH (l:Line)
            WHERE NOT (l)-[:OPERATED_BY]->()
            MERGE (l)-[:OPERATED_BY]->(op)
            """)
        except Exception:
            pass

    print("Ingest complete.")

# === GPT + schema introspection + validation/fallbacks ===

# Base system prompt (describes db labels/directions + examples). We'll append dynamic schema.
SYSTEM_PROMPT_BASE = """
You are a strict JSON-only translator. Convert a single natural-language question about the London Underground (TfL)
into exactly one JSON object with keys "cypher" and "params". Output MUST be only that JSON object and nothing else.

Database schema (labels & main props):
- Station(id,name,lat,lon,raw)
- StopPoint(id,name,commonName,lat,lon,modes,raw)
- Platform(id,name,raw)
- Line(id,name,mode,raw)
- FareZone(name)
- Operator(name)
- BikePoint(id,name,lat,lon,raw)
- Incident(id,statusSeverity,statusSeverityDescription,raw)
- Interchange(id,stationId,lines)
- Timetable(id,raw,nextArrivals)
- AccessibilityFeature(name)
- Facility(name)
- Landmark(id,name,category,lat,lon,raw)
- Geolocation(lat,lon)

Important RELATIONSHIPS and DIRECTIONS in this DB (canonical):
- (Station)-[:ON_LINE]->(Line)
- (StopPoint)-[:PART_OF]->(Station)
- (Station)-[:HAS_PLATFORM]->(Platform)
- (Station)-[:IN_ZONE]->(FareZone)
- (Station)-[:OPERATED_BY]->(Operator)
- (Station)-[:HAS_ACCESSIBILITY]->(AccessibilityFeature)
- (Station)-[:HAS_FACILITY]->(Facility)
- (Station)-[:HAS_INTERCHANGE]->(Interchange)
- (Line)-[:HAS_INCIDENT]->(Incident)
- (Station)-[:NEXT_TO]->(Station)
- (Station)-[:HAS_BIKEPOINT]->(BikePoint)
- (Station)-[:NEAR]->(Landmark)
- (Station)-[:LOCATED_AT]->(Geolocation)

Guidelines:
- Output JSON only: { "cypher": "<cypher string>", "params": { ... } }
- Use parameter placeholders ($param) for user-provided text.
- Use toLower(... ) CONTAINS to match names robustly.
- When matching a line, prefer:
    MATCH (l:Line) WHERE toLower(l.id)=toLower($q) OR toLower(l.name) CONTAINS toLower($q)
- For "stations on a line", use: MATCH (s:Station)-[:ON_LINE]->(l:Line) ...
- Limit results sensibly (LIMIT 200) unless user asks for everything.
- If a relation does not exist in the DB, prefer expressing equivalent logic using existing relations (e.g., treat a station that ->ON_LINE to >=2 lines as an interchange).

Examples (style):
User: Which stations are on the Northern line?
Output:
{"cypher":"MATCH (l:Line) WHERE toLower(l.id)=toLower($q) OR toLower(l.name) CONTAINS toLower($q) WITH l MATCH (s:Station)-[:ON_LINE]->(l) RETURN DISTINCT s.id AS id, s.name AS name ORDER BY s.name LIMIT 200","params":{"q":"northern"}}
"""

OPENAI_MODEL = "gpt-4"

# --- Schema introspection functions ---
def introspect_db_schema() -> Dict[str, set]:
    rels = set()
    labels = set()
    props = set()
    with driver.session() as s:
        try:
            for row in s.run("CALL db.relationshipTypes()"):
                try:
                    val = row.value()
                except Exception:
                    val = row[0]
                if val:
                    rels.add(str(val))
        except Exception:
            pass
        try:
            for row in s.run("CALL db.labels()"):
                try:
                    val = row.value()
                except Exception:
                    val = row[0]
                if val:
                    labels.add(str(val))
        except Exception:
            pass
        try:
            for row in s.run("CALL db.propertyKeys()"):
                try:
                    val = row.value()
                except Exception:
                    val = row[0]
                if val:
                    props.add(str(val))
        except Exception:
            pass
    return {"rels": rels, "labels": labels, "props": props}

_rel_re = re.compile(r"-\s*\[\s*(?:[A-Za-z0-9_]+?\s*:\s*)?`?([A-Za-z0-9_]+)`?\s*\]")

def extract_rel_types_from_cypher(cypher: str) -> set:
    return set(_rel_re.findall(cypher))

def request_gpt_for_cypher(question: str, system_prompt: str, model: str = OPENAI_MODEL, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": question}
        ],
        temperature=temperature,
        max_tokens=900
    )
    # robust extraction
    if isinstance(resp, dict):
        content = resp["choices"][0]["message"]["content"]
    else:
        content = resp.choices[0].message.content
    return content

def extract_json_from_text(s: str) -> Dict[str, Any]:
    m = re.search(r"(\{[\s\S]*\})", s)
    if not m:
        raise ValueError("No JSON found in model output")
    return json.loads(m.group(1))

def validate_and_autofix_cypher(original_cypher: str, schema: Dict[str, set]) -> Tuple[str, Optional[str]]:
    rels_in_query = extract_rel_types_from_cypher(original_cypher)
    missing = [r for r in rels_in_query if r not in schema["rels"]]
    if not missing:
        return original_cypher, None
    # special-case: HAS_INTERCHANGE missing
    if "HAS_INTERCHANGE" in missing:
        fix = (
            "// Auto-fixed: DB has no HAS_INTERCHANGE. Define interchange as station serving >=2 lines\n"
            "MATCH (s:Station)-[:ON_LINE]->(l:Line)\n"
            "WITH s, collect(DISTINCT l.id) AS linesServed\n"
            "WHERE size(linesServed) > 1\n"
            "UNWIND linesServed AS lineId\n"
            "MATCH (l2:Line {id: lineId})\n"
            "RETURN l2.id AS id, l2.name AS name, COUNT(DISTINCT s) AS interchange_count\n"
            "ORDER BY interchange_count DESC LIMIT 1"
        )
        return fix, f"Missing rel types: {missing}. Rewrote to count stations with >=2 lines as interchanges."
    # try swapping ON_LINE direction if it is present in query but wrong in DB
    cy = original_cypher
    if "ON_LINE" in rels_in_query:
        # replace occurrences of "(l:Line)<-[:ON_LINE]-(s:Station)" with "(s:Station)-[:ON_LINE]->(l:Line)"
        swapped = re.sub(r"\(\s*([^\)]+):Line\s*\)\s*<-\s*\[:ON_LINE\]\s*-\s*\(\s*([^\)]+):Station\s*\)", r"(\2:Station)-[:ON_LINE]->(\1:Line)", cy, flags=re.IGNORECASE)
        swapped = swapped.replace(")<-[:ON_LINE]-(", ")-[:ON_LINE]->(")
        if swapped != cy:
            return swapped, "Swapped ON_LINE direction to match canonical Station->ON_LINE->Line"
    # no auto-fix found
    return original_cypher, f"Missing rel types: {missing}; no auto-fix applied."

def translate_question_to_cypher(question: str) -> Tuple[str, Dict[str, Any]]:
    schema = introspect_db_schema()
    rel_list = sorted(list(schema["rels"]))
    label_list = sorted(list(schema["labels"]))
    dynamic_snippet = (
        "Database relationship types available: " + (", ".join(rel_list) if rel_list else "(none)") + "\n"
        "Labels available: " + (", ".join(label_list) if label_list else "(none)") + "\n\n"
        "If a relation requested by the user is not available above, do NOT invent it; prefer to express equivalent logic using existing relations (for example, treat a station with >=2 ON_LINE relations as an interchange)."
    )
    system_prompt = SYSTEM_PROMPT_BASE + "\n\n" + dynamic_snippet
    raw_out = request_gpt_for_cypher(question, system_prompt)
    try:
        data = extract_json_from_text(raw_out)
    except Exception:
        # retry with brief instruction
        followup = "Your last response did not contain the required JSON. Please respond exactly with a JSON object: {\"cypher\":\"...\",\"params\":{...}} and nothing else."
        raw2 = request_gpt_for_cypher(question + "\n\n" + followup, system_prompt)
        data = extract_json_from_text(raw2)
    cypher_gen = data.get("cypher", "")
    params = data.get("params", {}) or {}
    fixed, reason = validate_and_autofix_cypher(cypher_gen, schema)
    if reason:
        print("[AutoFix]", reason)
        print("[AutoFix] fixed cypher preview:\n", fixed)
    return fixed, params

# --- Execution + fallback runner ---
def run_cypher_and_fetch(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    with driver.session() as s:
        res = s.run(cypher, **(params or {}))
        return [r.data() for r in res]

def run_with_fallbacks_and_alternatives(cypher: str, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    tried = []
    # try original
    try:
        rows = run_cypher_and_fetch(cypher, params)
        tried.append(("original", cypher, rows))
        if rows:
            return rows, tried
    except Exception as e:
        tried.append(("original-error", cypher, str(e)))
    # swap ON_LINE direction if present
    if "ON_LINE" in cypher:
        alt = cypher.replace(")<-[:ON_LINE]-(", ")-[:ON_LINE]->(").replace("<-[:ON_LINE]-", "-[:ON_LINE]->")
        try:
            rows = run_cypher_and_fetch(alt, params)
            tried.append(("swap-on-line", alt, rows))
            if rows:
                return rows, tried
        except Exception as e:
            tried.append(("swap-error", alt, str(e)))
    # fallback: if interchange requested, run standard interchange count
    if re.search(r"interchang", cypher, re.IGNORECASE) or "HAS_INTERCHANGE" in cypher:
        fallback = (
            "MATCH (s:Station)-[:ON_LINE]->(l:Line)\n"
            "WITH s, collect(DISTINCT l.id) AS linesServed\n"
            "WHERE size(linesServed) > 1\n"
            "UNWIND linesServed AS lineId\n"
            "MATCH (l2:Line {id: lineId})\n"
            "RETURN l2.id AS id, l2.name AS name, COUNT(DISTINCT s) AS interchange_count\n"
            "ORDER BY interchange_count DESC LIMIT 5"
        )
        try:
            rows = run_cypher_and_fetch(fallback, {})
            tried.append(("fallback-interchange", fallback, rows))
            if rows:
                return rows, tried
        except Exception as e:
            tried.append(("fallback-error", fallback, str(e)))
    # last resort: return last tried (empty) + log
    return [], tried

# --- Interactive ask loop ---
def interactive_ask_loop():
    print("Interactive Ask — English (or Vietnamese). Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ")
        except KeyboardInterrupt:
            print("\nBye")
            return
        if not q or q.strip().lower() in ("exit", "quit"):
            print("Exit.")
            return
        try:
            print("Translating to Cypher via GPT-4 (with schema introspection)...")
            cypher, params = translate_question_to_cypher(q)
            print("Cypher:", cypher)
            print("Params:", json.dumps(params, ensure_ascii=False))
            print("Executing on Neo4j (with fallbacks)...")
            rows, tried = run_with_fallbacks_and_alternatives(cypher, params)
            print("Tried attempts (ordered):")
            for t in tried:
                tag, c, out = t
                if isinstance(out, list):
                    print(f"- {tag}: returned {len(out)} rows")
                else:
                    print(f"- {tag}: error -> {out}")
            print(f"RESULTS ({len(rows)} rows):")
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        except Exception as e:
            print("Error during ask:", e)
            traceback.print_exc()

# --- CLI entrypoint ---
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest TfL data into Neo4j")
    parser.add_argument("--ask", action="store_true", help="Start interactive ask loop (requires OPENAI_API_KEY)")
    parser.add_argument("--places-radius", type=int, default=200, help="radius (m) for Place lookup around a station")
    args = parser.parse_args()

    if args.ingest:
        ingest_all(radius_for_places=args.places_radius)
    if args.ask:
        interactive_ask_loop()
    if not args.ingest and not args.ask:
        parser.print_help()

if __name__ == "__main__":
    main()


    
#python graph_rag_tfl_full.py --ask
#Which line have the most interchange ?
