import urllib.request, json

payload = json.dumps({
    "profile": {
        "name": "Poojitha",
        "job_title": "Software Engineer",
        "experience_level": "entry",
        "years_experience": 1,
        "skills": ["python", "sql", "git"]
    },
    "num_questions": 3
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:5000/questions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
resp = urllib.request.urlopen(req, timeout=30)
data = json.loads(resp.read())

print("Demo mode:", data["demo_mode"])
print("Model:", data.get("model", "unknown"))
print("Questions generated:", len(data["questions"]))
print()
for i, q in enumerate(data["questions"], 1):
    print(f"Q{i}: {q['question']}")
    print(f"    Category: {q.get('category')} | Difficulty: {q.get('difficulty')}")
    print()
