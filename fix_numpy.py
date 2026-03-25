import json

NOTE = '# NumPy 2.0 removed np.float_ used by SMIRK/chumpy'

# ── Evaluation notebook ───────────────────────────────────────────────────────
eval_path = r'p:\SMIRK V2.0\milestone 2\SMIRK_Evaluation_PhD.ipynb'
OLD_EVAL = "sh('pip install -q iopath fvcore')"
NEW_EVAL = f"sh('pip install -q \"numpy<2.0\"')  {NOTE}\nsh('pip install -q iopath fvcore')"

with open(eval_path, 'r', encoding='utf-8') as f:
    raw = f.read()

if OLD_EVAL in raw:
    raw = raw.replace(OLD_EVAL, NEW_EVAL)
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write(raw)
    print('Fixed: Evaluation notebook')
else:
    print('No match in Evaluation notebook — checking content...')
    # find iopath in it
    idx = raw.find('iopath')
    print(repr(raw[max(0,idx-150):idx+100]))

# ── Lightning notebook — read and fix via JSON ─────────────────────────────────
lightning_path = r'p:\SMIRK V2.0\milestone 2\SMIRK_Inference_Lightning.ipynb'
OLD_LIT = 'pip install -q iopath fvcore'
NEW_LIT = f'pip install -q "numpy<2.0"  {NOTE}\npip install -q iopath fvcore'

with open(lightning_path, 'r', encoding='utf-8') as f:
    raw2 = f.read()

if OLD_LIT in raw2:
    raw2 = raw2.replace(OLD_LIT, NEW_LIT)
    with open(lightning_path, 'w', encoding='utf-8') as f:
        f.write(raw2)
    print('Fixed: Lightning notebook')
else:
    print('No match in Lightning notebook')
