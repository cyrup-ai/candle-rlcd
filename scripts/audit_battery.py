"""The usefulness audit's battery: 40 hand-written, unambiguous Jev-style cases (19 choice,
14 noul, 7 score) sent to a running server one at a time.

  python scripts/audit_battery.py [URL] [--json out.json]

laya as published scores 33/40 here (Score 3/7); see the audit in the project files.
"""
import json, sys, time, urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--json")]
URL = args[0] if args and not args[0].endswith(".json") else "http://127.0.0.1:8080"
JSON_OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

def C(ins, crit, exp): return ({"type": "choice", "instructions": ins, "criteria": crit}, exp)
def N(ins, exp): return ({"type": "noul", "instructions": ins}, exp)
def S(ins, levels, exp): return ({"type": "score", "instructions": ins, "criteria": levels}, exp)

TEAM = {"billing": "Payments, charges, refunds, invoices", "technical": "Bugs, errors, crashes, login problems", "shipping": "Deliveries, tracking, lost or damaged packages", "account": "Profile, email or password changes"}
SENT = {"positive": "Happy, satisfied", "negative": "Unhappy, angry, disappointed", "neutral": "Neither"}
URG = ["low", "medium", "high"]

CASES = [
 ("I was charged twice for my subscription this month, please refund one of the charges.", C("Which team handles this?", TEAM, "billing")),
 ("The app crashes every time I open the settings page on Android 14.", C("Which team handles this?", TEAM, "technical")),
 ("My package says delivered but it's not at my door. Tracking number 1Z999.", C("Which team handles this?", TEAM, "shipping")),
 ("How do I change the email address on my profile?", C("Which team handles this?", TEAM, "account")),
 ("I can't log in, it says invalid token after I reset my password.", C("Which team handles this?", TEAM, "technical")),
 ("Your invoice #4411 shows the wrong VAT number, please reissue it.", C("Which team handles this?", TEAM, "billing")),
 ("Absolutely love this blender, it crushed ice in seconds. Five stars!", C("What is the sentiment?", SENT, "positive")),
 ("Worst purchase ever. Broke after two days and support ignored me.", C("What is the sentiment?", SENT, "negative")),
 ("The meeting is moved to 3pm on Thursday in room B.", C("What is the sentiment?", SENT, "neutral")),
 ("I'm so disappointed, the hotel room was dirty and smelled of smoke.", C("What is the sentiment?", SENT, "negative")),
 ("Thank you so much, the replacement arrived early and works perfectly!", C("What is the sentiment?", SENT, "positive")),
 ("Our production database is down and no customer can check out. We are losing thousands per minute!", S("How urgent is this?", URG, "high")),
 ("Whenever you get a chance, could you update the logo in the footer? No rush.", S("How urgent is this?", URG, "low")),
 ("The server is on fire, literally, smoke coming out of rack 4. Evacuating now.", S("How urgent is this?", URG, "high")),
 ("Small typo on the About page: 'recieve' should be 'receive'. Not urgent at all.", S("How urgent is this?", URG, "low")),
 ("I want my money back for the broken headphones.", N("Does the customer want a refund?", True)),
 ("Just wanted to say the headphones sound great, thanks!", N("Does the customer want a refund?", False)),
 ("Can you please cancel my account effective immediately?", N("Does the user want to cancel their account?", True)),
 ("What are your opening hours on Sunday?", N("Does the user want to cancel their account?", False)),
 ("I'm going to find where you live and hurt you.", N("Does this message contain a threat of violence?", True)),
 ("Have a lovely weekend, see you Monday!", N("Does this message contain a threat of violence?", False)),
 ("Hi, I'm John Smith, my phone is 555-201-3344.", N("Does the text contain a phone number?", True)),
 ("The weather in Paris is sunny today.", N("Does the text contain a phone number?", False)),
 ("Paris is the capital of France.", N("Is this statement factually correct?", True)),
 ("The sun orbits the earth once a day.", N("Is this statement factually correct?", False)),
 ("Bonjour, je voudrais réserver une table pour deux ce soir.", C("What language is this text in?", {"english": "English", "french": "French", "german": "German", "spanish": "Spanish"}, "french")),
 ("Guten Morgen, wie geht es Ihnen heute?", C("What language is this text in?", {"english": "English", "french": "French", "german": "German", "spanish": "Spanish"}, "german")),
 ("def add(a, b):\n    return a + b", C("What kind of content is this?", {"code": "Source code", "poem": "Poetry", "recipe": "Cooking recipe", "email": "An email"}, "code")),
 ("Preheat oven to 180C. Mix flour, sugar and eggs, bake for 25 minutes.", C("What kind of content is this?", {"code": "Source code", "poem": "Poetry", "recipe": "Cooking recipe", "email": "An email"}, "recipe")),
 ("Lakers beat the Celtics 112-104 behind 38 points from LeBron.", C("Which topic is this?", {"sports": "Sports", "politics": "Politics", "tech": "Technology", "health": "Health"}, "sports")),
 ("The Senate passed the budget bill 52-48 after a late-night vote.", C("Which topic is this?", {"sports": "Sports", "politics": "Politics", "tech": "Technology", "health": "Health"}, "politics")),
 ("A new study links daily walking to lower risk of heart disease.", C("Which topic is this?", {"sports": "Sports", "politics": "Politics", "tech": "Technology", "health": "Health"}, "health")),
 ("Apple unveiled a new M5 chip with a faster neural engine.", C("Which topic is this?", {"sports": "Sports", "politics": "Politics", "tech": "Technology", "health": "Health"}, "tech")),
 ("user: my order hasn't arrived\nagent: I've issued a full refund, you'll see it in 3 days\nuser: great, thanks so much!", N("Was the customer's issue resolved?", True)),
 ("user: my order hasn't arrived\nagent: please wait another week\nuser: this is ridiculous, I'm disputing the charge with my bank", N("Was the customer's issue resolved?", False)),
 ("Congratulations! You've WON a $1000 gift card. Click http://bit.ly/xyz to claim now!!!", N("Is this message spam?", True)),
 ("Hi team, attaching the Q3 report for tomorrow's review. Thanks, Maria", N("Is this message spam?", False)),
 ("The product is okay. It works but the battery life is mediocre.", S("Rate the review from 1 to 5 stars", ["1", "2", "3", "4", "5"], "3")),
 ("Terrible. Arrived broken, refund refused. Never again.", S("Rate the review from 1 to 5 stars", ["1", "2", "3", "4", "5"], "1")),
 ("Perfect in every way, exceeded all expectations, buying another!", S("Rate the review from 1 to 5 stars", ["1", "2", "3", "4", "5"], "5")),
]

def post(body):
    req = urllib.request.Request(URL + "/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read()), time.time() - t

by = {}
ok = 0
rows = []
for state, (q, exp) in CASES:
    out, dt = post({"state": state, "model": "jev-latest", "questions": {"q": q}})
    a = out["answers"]["q"]
    if q["type"] == "choice": got = a["choice"]
    elif q["type"] == "score": got = q["criteria"][max(range(len(a["probabilities"])), key=lambda i: a["probabilities"][str(i)])]
    else: got = a["noul"] >= 0.5
    good = got == exp
    ok += good
    by.setdefault(q["type"], [0, 0]); by[q["type"]][0] += good; by[q["type"]][1] += 1
    rows.append((good, q["type"], state[:60].replace("\n", " "), exp, got, a.get("confidence", a.get("noul")), round(dt * 1000)))
for r in rows: print(("OK  " if r[0] else "MISS"), *r[1:], sep=" | ")
print(f"\n{ok}/{len(CASES)} correct", {k: f"{v[0]}/{v[1]}" for k, v in by.items()})
if JSON_OUT:
    with open(JSON_OUT, "w") as f:
        json.dump({"correct": ok, "total": len(CASES), "by_type": {k: {"correct": v[0], "total": v[1]} for k, v in by.items()},
                   "misses": [{"type": r[1], "state": r[2], "expected": r[3], "got": r[4]} for r in rows if not r[0]],
                   "mean_latency_ms": sum(r[6] for r in rows) / len(rows)}, f, indent=1)
