import requests, json
url = 'http://127.0.0.1:8000/query'
payload = {
    "question": "What is Newton's second law?",
    "top_k": 5,
}
import time
try:
    start = time.time()
    r = requests.post(url, json=payload, timeout=180)
    elapsed = time.time() - start
    print('STATUS', r.status_code)
    print('ELAPSED', round(elapsed, 2), 's')
    print(r.text)
except Exception as e:
    print('ERROR', e)
