import os, pickle, sys
from pprint import pprint

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for name in ['train','val']:
    p = os.path.join(base, 'dataset', f'{name}.pkl')
    print('='*80)
    print('FILE', name, p, 'exists=', os.path.exists(p))
    if not os.path.exists(p):
        continue
    with open(p, 'rb') as f:
        d = pickle.load(f)
    try:
        n = len(d)
    except Exception as e:
        n = f'? {e}'
    print('LEN', name, n, 'TYPE', type(d))
    # Peek first 5
    try:
        items = list(d[:5]) if hasattr(d, '__getitem__') else [next(iter(d))]
    except Exception as e:
        print('ITER ERROR', e)
        items = []
    for i, it in enumerate(items):
        print(f'--- ITEM {i} TYPE {type(it)}')
        if isinstance(it, dict):
            keys = list(it.keys())
            print('keys:', keys)
            img = it.get('image', it.get('img', it.get('path')))
            tex = it.get('latex', it.get('formula'))
            if img is not None:
                abs_img = img if os.path.isabs(img) else os.path.abspath(os.path.join(base, 'dataset', img))
                print('image path:', img, '->', abs_img, 'exists=', os.path.exists(abs_img))
            print('latex snippet:', (tex or '')[:120])
        else:
            try:
                img, tex = it
            except Exception:
                pprint(it)
                continue
            if isinstance(img, str):
                abs_img = img if os.path.isabs(img) else os.path.abspath(os.path.join(base, 'dataset', img))
                print('image path:', img, '->', abs_img, 'exists=', os.path.exists(abs_img))
            print('latex snippet:', (tex or '')[:120])

print('='*80)
print('Done')
