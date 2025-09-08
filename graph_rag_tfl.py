"""
graph_rag_tfl.py
Updated version (2025-09-06) with fixes:
- Fix TfL mode names (normalize 'elizabeth' -> 'elizabeth-line')
- Fetch valid modes from TfL metadata and normalize inputs
- Log TfL response bodies on error for easier debugging
- Use app_id + app_key if provided in .env
- Use Neo4j native CREATE FULLTEXT INDEX DDL (with fallback)
- Keep GPT-4 -> Cypher agent (read-only) and ingestion pipeline
"""

import os
import time
import json
import re
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
from neo4j import GraphDatabase
import openai
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
load_dotenv()
TFL_APP_KEY = os.getenv("TFL_APP_KEY")
TFL_APP_ID = os.getenv("TFL_APP_ID")  # optional, sometimes required for older keys
TFL_BASE = "https://api.tfl.gov.uk"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TFL_APP_KEY:
    raise SystemExit("Missing TFL_APP_KEY in environment.")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set; GPT->Cypher agent won't work until set.")

openai.api_key = OPENAI_API_KEY

# limit how many lines we ingest by default (to avoid hammering API while testing)
DEFAULT_LINE_LIMIT = 10

# -------------------------
# TfL client (small wrapper)
# -------------------------
class TfLClient:
    def __init__(self, app_key: str, base_url: str = TFL_BASE, session: Optional[requests.Session]=None, app_id: Optional[str]=None):
        self.base = base_url.rstrip("/")
        self.app_key = app_key
        self.app_id = app_id
        self.s = session or requests.Session()

    def _get(self, path: str, params: Dict[str,Any] = None, backoff=3) -> Any:
        if params is None:
            params = {}
        # attach credentials; include app_id if present
        params["app_key"] = self.app_key
        if self.app_id:
            params["app_id"] = self.app_id
        url = f"{self.base}{path}"
        for attempt in range(3):
            try:
                r = self.s.get(url, params=params, timeout=20)
            except Exception as e:
                # network error -> backoff
                if attempt < 2:
                    time.sleep(backoff * (attempt+1))
                    continue
                raise
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    # not JSON; return text
                    return r.text
            elif r.status_code in (429, 503):
                time.sleep(backoff * (attempt+1))
                continue
            else:
                # print response body to help debug (TfL often returns JSON message)
                try:
                    print(f"TfL API error {r.status_code} for {url} -> {r.text}")
                except Exception:
                    print(f"TfL API error status: {r.status_code} for {url}")
                r.raise_for_status()
        r.raise_for_status()

    def get_valid_modes(self) -> List[str]:
        """
        Attempt to fetch official modes list from TfL metadata endpoints.
        Falls back to a sensible default list.
        """
        try:
            resp = self._get("/StopPoint/Meta/Modes")
            modes = []
            if isinstance(resp, list):
                for r in resp:
                    if isinstance(r, dict):
                        if "modeName" in r:
                            modes.append(r["modeName"])
                        elif "id" in r:
                            modes.append(r["id"])
                    elif isinstance(r, str):
                        modes.append(r)
            # de-duplicate while preserving order
            return list(dict.fromkeys(modes))
        except Exception as e:
            # fallback
            return ["tube", "overground", "dlr", "elizabeth-line", "national-rail", "tram", "bus"]

    def get_lines_by_modes(self, modes: List[str]) -> List[Dict]:
        # ensure we use valid mode ids as per TfL metadata
        valid = set(self.get_valid_modes())
        normalized = []
        for m in modes:
            if not m:
                continue
            mm = m.strip().lower()
            # common alias fixes
            if mm == "elizabeth":
                mm = "elizabeth-line"
            if mm == "overground":
                mm = "overground"
            # if exact match to known valid mode
            if mm in valid:
                normalized.append(mm)
            else:
                # fuzzy match: try to find a valid mode starting/containing mm
                found = False
                for v in valid:
                    if v.startswith(mm) or mm.startswith(v) or mm in v or v in mm:
                        normalized.append(v)
                        found = True
                        break
                if not found:
                    # include original normalized token (let API decide), but log
                    print(f"[TfLClient] Warning: mode '{m}' not recognized among metadata; including '{mm}' in request.")
                    normalized.append(mm)
        modes_csv = ",".join(dict.fromkeys(normalized))
        return self._get(f"/Line/Mode/{modes_csv}/Status")

    def get_line_stop_points(self, line_id: str) -> List[Dict]:
        return self._get(f"/Line/{line_id}/StopPoints")

    def get_line_route_sequence(self, line_id: str, direction: str="outbound") -> Dict:
        return self._get(f"/Line/{line_id}/Route/Sequence/{direction}", params={"serviceTypes":"Regular","excludeCrowding":"true"})

    def get_stop_point(self, stop_id: str) -> Dict:
        return self._get(f"/StopPoint/{stop_id}")

    def get_stop_arrivals(self, stop_id: str) -> List[Dict]:
        return self._get(f"/StopPoint/{stop_id}/Arrivals")

    def search_stop_points(self, query: str, modes: List[str]=None, maxResults:int=25) -> Dict:
        params = {"query": query, "maxResults": maxResults}
        if modes:
            params["modes"] = ",".join(modes)
        return self._get("/StopPoint/Search", params=params)

# -------------------------
# Neo4j client / schema + upsert helpers
# -------------------------
class Neo4jKG:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_write(self, cypher: str, params: dict = None):
        params = params or {}
        with self.driver.session() as session:
            return session.execute_write(lambda tx: tx.run(cypher, **params).data())

    def execute_read(self, cypher: str, params: dict = None):
        params = params or {}
        with self.driver.session() as session:
            return session.execute_read(lambda tx: tx.run(cypher, **params).data())

    def create_schema(self):
        """
        Create constraints and a fulltext index for retrieval.
        Node types: Station, Line, Platform, Service, Timetable, Connection,
                    FareZone, Operator, Depot, AccessibilityFeature, Incident, Ridership
        Uses native DDL for fulltext where supported.
        """
        statements = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Station) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Line) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Platform) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Service) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Timetable) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Connection) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:FareZone) REQUIRE n.zone_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Operator) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Depot) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:AccessibilityFeature) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Incident) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Ridership) REQUIRE n.id IS UNIQUE;"
        ]
        for st in statements:
            try:
                self.execute_write(st)
            except Exception as e:
                # some neo4j versions differ in syntax/privileges; log and continue
                print("Schema stmt failed (may be permissions/version):", st, str(e))

        # create a combined fulltext index using native DDL (Neo4j 4.3+)
        try:
            create_fulltext = """
            CREATE FULLTEXT INDEX IF NOT EXISTS kg_fulltext
            FOR (n:Station|Line|Platform|Incident|AccessibilityFeature|Operator|Depot|Service|Timetable|FareZone|Ridership|Connection)
            ON EACH [n.name, n.description, n.notes, n.summary];
            """
            self.execute_write(create_fulltext)
        except Exception as e:
            # native DDL might not be supported on older Neo4j versions; log and continue
            print("Fulltext index creation skipped or failed (native DDL attempt):", e)

    # Upsert helper examples
    def upsert_station(self, st: Dict):
        cy = """
        MERGE (s:Station {id: $id})
        SET s.name=$name,
            s.lat=$lat,
            s.lon=$lon,
            s.commonName=$commonName,
            s.naptanId=$naptanId,
            s.modeName = $modeName,
            s.raw = $raw
        RETURN s.id as id;
        """
        props = {
            "id": st.get("id") or st.get("naptanId") or st.get("icsCode") or st.get("stationId"),
            "name": st.get("name") or st.get("commonName"),
            "lat": st.get("lat") or st.get("lat"),
            "lon": st.get("lon") or st.get("lon"),
            "commonName": st.get("commonName"),
            "naptanId": st.get("naptanId"),
            "modeName": ",".join(st.get("modes", [])) if st.get("modes") else st.get("modeName"),
            "raw": json.dumps(st)
        }
        try:
            return self.execute_write(cy, props)
        except Exception as e:
            print("Upsert station failed:", e, props.get("id"))
            return None

    def upsert_line(self, line: Dict):
        cy = """
        MERGE (l:Line {id: $id})
        SET l.name = $name,
            l.modeName = $modeName,
            l.lineStatuses = $lineStatuses,
            l.raw = $raw
        RETURN l.id as id;
        """
        props = {
            "id": line.get("id") or line.get("lineId"),
            "name": line.get("name"),
            "modeName": line.get("modeName"),
            "lineStatuses": json.dumps(line.get("lineStatuses", [])),
            "raw": json.dumps(line)
        }
        try:
            return self.execute_write(cy, props)
        except Exception as e:
            print("Upsert line failed:", e, props.get("id"))
            return None

    def connect_station_line(self, station_id: str, line_id: str):
        cy = """
        MATCH (s:Station {id:$station_id}), (l:Line {id:$line_id})
        MERGE (s)-[:ON_LINE]->(l)
        """
        self.execute_write(cy, {"station_id": station_id, "line_id": line_id})

    def upsert_platform(self, platform: Dict, station_id: str):
        cy = """
        MERGE (p:Platform {id: $id})
        SET p.platformName=$platformName, p.platformNumber=$platformNumber, p.raw=$raw
        WITH p
        MATCH (s:Station {id:$station_id})
        MERGE (s)-[:HAS_PLATFORM]->(p)
        """
        props = {
            "id": platform.get("id") or platform.get("platformId") or f"{station_id}:{platform.get('platformName','')}",
            "platformName": platform.get("platformName") or platform.get("name"),
            "platformNumber": platform.get("platformNumber"),
            "raw": json.dumps(platform),
            "station_id": station_id
        }
        try:
            self.execute_write(cy, props)
        except Exception as e:
            print("Upsert platform failed:", e, props.get("id"))

    def upsert_incident(self, inc: Dict, affected_entities: List[Dict]=None):
        cy = """
        MERGE (i:Incident {id:$id})
        SET i.title=$title, i.type=$type, i.startDate=$startDate, i.endDate=$endDate, i.severity=$severity, i.raw=$raw
        """
        props = {
            "id": inc.get("id") or inc.get("created"),
            "title": inc.get("title") or inc.get("description"),
            "type": inc.get("type") or inc.get("category"),
            "startDate": inc.get("startDate"),
            "endDate": inc.get("endDate"),
            "severity": inc.get("severity"),
            "raw": json.dumps(inc)
        }
        self.execute_write(cy, props)
        # connect to affected nodes (stations/lines)
        if affected_entities:
            for ent in affected_entities:
                if ent.get("type") == "Station" or ent.get("entityType") == "StopPoint":
                    sid = ent.get("id") or ent.get("naptanId")
                    if sid:
                        self.execute_write("MATCH (s:Station {id:$sid}), (i:Incident {id:$iid}) MERGE (s)-[:AFFECTED_BY]->(i)",
                                           {"sid": sid, "iid": props["id"]})
                if ent.get("type") == "Line" or ent.get("entityType") == "Line":
                    lid = ent.get("id")
                    if lid:
                        self.execute_write("MATCH (l:Line {id:$lid}), (i:Incident {id:$iid}) MERGE (l)-[:AFFECTED_BY]->(i)",
                                           {"lid": lid, "iid": props["id"]})

    def upsert_accessibility_feature(self, feat: Dict, station_id: str):
        cy = """
        MERGE (a:AccessibilityFeature {id:$id})
        SET a.feature_type=$feature_type, a.notes=$notes, a.raw=$raw
        WITH a
        MATCH (s:Station {id:$station_id})
        MERGE (s)-[:HAS_ACCESS_FEATURE]->(a)
        """
        props = {
            "id": feat.get("id") or f"{station_id}:access:{feat.get('type', 'unknown')}",
            "feature_type": feat.get("type") or feat.get("featureType"),
            "notes": feat.get("notes") or feat.get("description"),
            "raw": json.dumps(feat),
            "station_id": station_id
        }
        try:
            self.execute_write(cy, props)
        except Exception as e:
            print("Upsert access feature failed:", e, props.get("id"))

# -------------------------
# Ingest pipeline
# -------------------------
def ingest_tfl_to_neo4j(tfl_client: TfLClient, kg: Neo4jKG, line_limit:int = DEFAULT_LINE_LIMIT):
    """
    Ingest a subset of TfL lines & their stops/routes, create nodes & relationships.
    """
    print("Creating schema (constraints & indexes)...")
    kg.create_schema()

    print("Fetching lines (modes: tube, overground, dlr, elizabeth-line)...")
    modes = ["tube","overground","dlr","elizabeth-line"]
    raw_lines = tfl_client.get_lines_by_modes(modes)
    if not isinstance(raw_lines, list):
        raw_lines = [raw_lines]
    print(f"Found {len(raw_lines)} lines; ingesting up to {line_limit} lines.")

    for line in tqdm(raw_lines[:line_limit], desc="Lines"):
        kg.upsert_line(line)
        line_id = line.get("id")
        # fetch stop points for the line
        try:
            stops = tfl_client.get_line_stop_points(line_id)
            for st in stops:
                kg.upsert_station(st)
                station_id = st.get("id") or st.get("naptanId")
                if station_id:
                    kg.connect_station_line(station_id, line_id)
            # route sequences (inbound + outbound) — useful to create Timetable/ServicePattern nodes
            for direction in ("inbound","outbound"):
                try:
                    seq = tfl_client.get_line_route_sequence(line_id, direction=direction)
                    tt_id = f"{line_id}:{direction}"
                    cy = """
                    MERGE (tt:Timetable {id:$id})
                    SET tt.line=$line, tt.direction=$direction, tt.raw=$raw
                    """
                    kg.execute_write(cy, {"id": tt_id, "line": line_id, "direction": direction, "raw": json.dumps(seq)})
                    kg.execute_write("MATCH (l:Line {id:$line}), (tt:Timetable {id:$ttid}) MERGE (l)-[:HAS_ROUTE]->(tt)",
                                     {"line": line_id, "ttid": tt_id})
                    olr = seq.get("orderedLineRoutes") or []
                    seq_stops = []
                    for route in olr:
                        stops_seq = route.get("stopPointSequences") or []
                        for sseq in stops_seq:
                            for stop in sseq.get("stopPoint", []):
                                sid = stop.get("id")
                                if sid:
                                    kg.upsert_station({"id": sid, "name": stop.get("name"), "modes":[line.get("modeName")]})
                                    seq_stops.append(sid)
                    # create adjacency relationships pairwise
                    for a,b in zip(seq_stops, seq_stops[1:]):
                        kg.execute_write("MATCH (s1:Station {id:$a}), (s2:Station {id:$b}) MERGE (s1)-[:NEXT_ON_LINE {line:$line}]->(s2)",
                                         {"a":a,"b":b,"line":line_id})
                except Exception as e:
                    print(f"Route sequence fetch failed for {line_id} {direction}: {e}")
            # optionally fetch live arrivals for the first few stations
            for st in (stops[:3] if stops else []):
                sid = st.get("id")
                if sid:
                    try:
                        arrivals = tfl_client.get_stop_arrivals(sid)
                        kg.execute_write("MATCH (s:Station {id:$sid}) SET s.last_arrivals = $arr", {"sid": sid, "arr": json.dumps(arrivals)})
                    except Exception as e:
                        print("Arrivals fetch failed for", sid, e)
        except Exception as e:
            print("Failed to fetch stop points for line:", line_id, e)

    print("Ingest complete (partial). You can extend ingestion to Operators, Depots, Accessibility full data, Ridership CSV import, etc.")
    print("Reminder: some node types (FareZone, Ridership) may need external datasets or TfL-specific endpoints; ingestion scaffolding is provided.")

# -------------------------
# GPT-4 -> Cypher Agent
# -------------------------
CYHER_SANDBOX_FORBIDDEN = [
    r"\bCREATE\b", r"\bMERGE\b", r"\bDELETE\b", r"\bSET\b", r"\bREMOVE\b",
    r"\bDROP\b", r"\bCALL\b", r"\bapoc\b", r"\bdbms\b", r"\bSYSTEM\b"
]
CYTHER_ALLOW_START = [r"^\s*MATCH\b", r"^\s*WITH\b", r"^\s*OPTIONAL\s+MATCH\b", r"^\s*RETURN\b", r"^\s*CALL db.index.fulltext.queryNodes"]

def validate_cypher_readonly(cypher: str) -> bool:
    """Ensure cypher is read-only (no schema changes, no DB-procedures)."""
    up = cypher.upper()
    for pat in CYHER_SANDBOX_FORBIDDEN:
        if re.search(pat, up, flags=re.IGNORECASE):
            return False
    # must contain MATCH or WITH or RETURN or fulltext call
    for pat in CYTHER_ALLOW_START:
        if re.search(pat, cypher, flags=re.IGNORECASE):
            return True
    # if nothing matched, reject
    return False

# def nl_to_cypher_via_gpt4(question: str, schema_doc: str, examples: List[Dict[str,str]]=None) -> str:
#     """
#     Ask GPT-4 to produce a read-only Cypher query.
#     - schema_doc: small textual description of node labels + key properties (we supply below)
#     - examples: optional list of {nl:..., cypher:...} to few-shot
#     Returns: cypher string (or raises)
#     """
#     if not OPENAI_API_KEY:
#         raise RuntimeError("OPENAI_API_KEY not set in environment.")

#     system = (
#         "You are an expert assistant that *translates natural-language queries about the London rail graph* "
#         "into read-only Neo4j Cypher queries. IMPORTANT: produce ONLY a single Cypher query (no explanation). "
#         "The query must be read-only: use MATCH / OPTIONAL MATCH / WITH / RETURN only. "
#         "Do NOT output CREATE/MERGE/DELETE/SET/CALL/apoc/dbms or any side-effect or admin commands. "
#         "Return the Cypher text only in a single code block if possible. If parameters are required, use $param style."
#     )

#     user_prompt = f"Schema:\n{schema_doc}\n\nUser question: {question}\n\nProduce a single read-only Cypher query."
#     messages = [{"role":"system","content":system}, {"role":"user","content":user_prompt}]
#     if examples:
#         for ex in examples:
#             messages.append({"role":"user","content":f"Example NL: {ex['nl']}\nExample Cypher: {ex['cypher']}"})

#     resp = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=messages,
#         temperature=0.0,
#         max_tokens=600
#     )
#     cypher = resp["choices"][0]["message"]["content"].strip()
#     # if wrapped in triple backticks, extract
#     m = re.search(r"```(?:cypher)?\n(.+?)```", cypher, flags=re.S|re.I)
#     if m:
#         cy = m.group(1).strip()
#     else:
#         cy = cypher
#     # validate
#     if not validate_cypher_readonly(cy):
#         raise ValueError("Generated Cypher did not pass read-only validation. Output:\n" + cy)
#     return cy


def nl_to_cypher_via_gpt4(question: str, schema_doc: str, examples: List[Dict[str,str]]=None) -> str:
    """
    Backwards- and forwards-compatible wrapper to call OpenAI chat:
    - supports new openai >=1.0.0 (openai.OpenAI client)
    - supports older openai (openai.ChatCompletion.create)
    Returns a single read-only Cypher string (validated).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    system = (
        "You are an expert assistant that translates natural-language queries about the London rail graph "
        "into read-only Neo4j Cypher queries. IMPORTANT: produce ONLY a single Cypher query (no explanation). "
        "The query must be read-only: use MATCH / OPTIONAL MATCH / WITH / RETURN only. "
        "Do NOT output CREATE/MERGE/DELETE/SET/CALL/apoc/dbms or any side-effect or admin commands. "
        "Return the Cypher text only. If parameters are required, use $param style."
    )

    user_prompt = f"Schema:\n{schema_doc}\n\nUser question: {question}\n\nProduce a single read-only Cypher query."
    messages = [{"role":"system","content":system}, {"role":"user","content":user_prompt}]
    if examples:
        for ex in examples:
            messages.append({"role":"user","content":f"Example NL: {ex['nl']}\nExample Cypher: {ex['cypher']}"})

    # Try new OpenAI client first (openai >= 1.0.0)
    try:
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.0,
                max_tokens=600
            )
            # new client returns objects; extract robustly
            if hasattr(resp, "choices"):
                # resp.choices[0].message.content in new client
                content = getattr(resp.choices[0].message, "content", None)
                if content is None:
                    # fallback if dict-like
                    content = resp.choices[0]["message"]["content"]
            else:
                # fallback to dict-like
                content = resp["choices"][0]["message"]["content"]
        else:
            # fallback to old API (openai < 1.0)
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.0,
                max_tokens=600
            )
            content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        # re-raise with context for debugging
        raise RuntimeError(f"OpenAI call failed: {e}")

    cypher_text = content.strip()
    # extract inside triple backticks if present
    m = re.search(r"```(?:cypher)?\n(.+?)```", cypher_text, flags=re.S|re.I)
    if m:
        cy = m.group(1).strip()
    else:
        cy = cypher_text

    if not validate_cypher_readonly(cy):
        raise ValueError("Generated Cypher did not pass read-only validation. Output:\n" + cy)
    return cy


def execute_and_format(kg: Neo4jKG, cypher: str) -> List[Dict[str,Any]]:
    """
    Execute read-only cypher and return results as list-of-dicts.
    """
    try:
        rows = kg.execute_read(cypher)
        return rows
    except Exception as e:
        raise

# -------------------------
# Small schema doc for GPT prompt (keeps GPT aware of node labels & key properties)
# -------------------------
SCHEMA_DOC = """
Node labels and key properties (Triples used in the KG):
- Station: id (string, TfL StopPoint id), name, lat, lon, naptanId, modeName
- Line: id (string), name, modeName, lineStatuses (json)
- Platform: id, platformName, platformNumber
- Service: id, line_id, origin, destination, scheduled_departure_time, status
- Timetable: id, line, direction, raw (service pattern)
- Connection: id, source_station_id, target_station_id, walking_time
- FareZone: zone_id, name
- Operator: id, name
- Depot: id, name, capacity
- AccessibilityFeature: id, feature_type, notes
- Incident: id, title, type, startDate, endDate, severity
- Ridership: id, station_id, period_start, period_end, count
Relationships:
- (s:Station)-[:ON_LINE]->(l:Line)
- (s:Station)-[:HAS_PLATFORM]->(p:Platform)
- (l:Line)-[:HAS_ROUTE]->(tt:Timetable)
- (s1:Station)-[:CONNECTED_TO {walking_time}]->(s2:Station)
- (s:Station)-[:HAS_ACCESS_FEATURE]->(a:AccessibilityFeature)
- (s|l)-[:AFFECTED_BY]->(i:Incident)
- (operator)-[:OPERATES]->(line)
- (station)-[:IN_ZONE]->(farezone)
Use these labels & properties to build MATCH queries.
"""

# -------------------------
# CLI
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python graph_rag_tfl.py [ingest|ask] [--line-limit N] [\"question text\"]")
        return
    cmd = sys.argv[1]
    tfl = TfLClient(TFL_APP_KEY, app_id=TFL_APP_ID)
    kg = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    if cmd == "ingest":
        line_limit = DEFAULT_LINE_LIMIT
        # optional flag
        if "--line-limit" in sys.argv:
            i = sys.argv.index("--line-limit")
            if i+1 < len(sys.argv):
                try:
                    line_limit = int(sys.argv[i+1])
                except:
                    pass
        ingest_tfl_to_neo4j(tfl, kg, line_limit)
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Provide question text in quotes: python graph_rag_tfl.py ask \"Where is ...\"")
            return
        question = " ".join(sys.argv[2:])
        print("Translating to Cypher via GPT-4...")
        try:
            cy = nl_to_cypher_via_gpt4(question, SCHEMA_DOC,
                                      examples=[
                                          {"nl":"List stations on the Central line with step-free access",
                                           "cypher":"MATCH (l:Line {name:'Central'})<-[:ON_LINE]-(s:Station)-[:HAS_ACCESS_FEATURE]->(a:AccessibilityFeature) WHERE toLower(a.feature_type) CONTAINS 'step-free' RETURN s.name AS station, s.id AS id LIMIT 200"}
                                      ])
            print("Generated Cypher:\n", cy)
        except Exception as e:
            print("Failed to generate Cypher:", e)
            return
        # safety last check
        if not validate_cypher_readonly(cy):
            print("Rejected unsafe cypher.")
            return
        # execute
        try:
            rows = execute_and_format(kg, cy)
            print("Results:", json.dumps(rows, indent=2, ensure_ascii=False))
        except Exception as e:
            print("Query execution failed:", e)
    else:
        print("Unknown command:", cmd)

    kg.close()

if __name__ == "__main__":
    main()


