import requests

resp = requests.post(
    'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1',
    data={'username': 'sunny70361@gmail.com', 'password': 'OmNamoNarayana@99'},
    allow_redirects=True,
    timeout=20,
    stream=True
)
print('Status:', resp.status_code)
print('Content-Type:', resp.headers.get('Content-Type'))
print('Content-Length:', resp.headers.get('Content-Length'))

if 'html' in resp.headers.get('Content-Type', ''):
    import re
    text = resp.text[:3000]
    # look for any message divs
    import re
    found = re.findall(r'<(?:div|p|span)[^>]*>\s*([^<]{10,150})\s*</', text)
    for f in found[:10]:
        f = f.strip()
        if f:
            print('Page text:', f)
