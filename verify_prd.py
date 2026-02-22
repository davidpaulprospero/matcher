import json

with open(r'D:\_Projects\voiceover-matcher-subtitle\scripts\ralph\state\prd.json', 'r') as f:
    d = json.load(f)

stories = d['userStories']
print(f"US-101-001 passes: {stories[0]['passes']}")

generated = [s for s in stories if s['id'].startswith('US-101-') and s['id'] != 'US-101-001']
print(f"Generated stories count: {len(generated)}")

all_false = all(s['passes'] == False for s in generated)
print(f"All generated stories have passes=false: {all_false}")

for s in generated:
    print(f"  {s['id']}: {s['title'][:50]}...")
