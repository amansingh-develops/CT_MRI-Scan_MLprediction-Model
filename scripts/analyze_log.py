import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

logpath = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\training log\livertumor-model (1).log'
lines = open(logpath, encoding='utf-8', errors='replace').readlines()

pattern = re.compile(r'Epoch \d|BEST|improvement|checkpoint|NaN|Inf grad|recovery|Warmup|RESUME|weights_only|Loaded|Grad  \||Train \||Val   \|')

for i, l in enumerate(lines):
    if pattern.search(l):
        print(f'{i+1}: {l.strip()}')
