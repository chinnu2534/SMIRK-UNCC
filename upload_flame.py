"""
Run this after downloading FLAME2020.zip manually from https://flame.is.tue.mpg.de
Usage: python upload_flame.py  (place FLAME2020.zip in same folder first)
"""
from huggingface_hub import HfApi
import os

TOKEN = 'YOUR_HF_TOKEN_HERE'
HF_USER = 'abhinavvelaga'
FLAME_ZIP = r'p:\SMIRK V2.0\FLAME2020.zip'   # <-- adjust if needed

api = HfApi(token=TOKEN)

# Create a private dataset repo to store the model file
try:
    api.create_repo(
        repo_id=f'{HF_USER}/smirk-assets',
        repo_type='dataset',
        private=True,
        exist_ok=True
    )
    print('Created private dataset repo: smirk-assets')
except Exception as e:
    print('Repo exists or error:', e)

# Upload FLAME2020.zip
print('Uploading FLAME2020.zip...')
api.upload_file(
    path_or_fileobj=FLAME_ZIP,
    path_in_repo='FLAME2020.zip',
    repo_id=f'{HF_USER}/smirk-assets',
    repo_type='dataset',
)
print('Upload complete!')
print(f'\nFLAME2020.zip is now at:')
print(f'https://huggingface.co/datasets/{HF_USER}/smirk-assets/resolve/main/FLAME2020.zip')
