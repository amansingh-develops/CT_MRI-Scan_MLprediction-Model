import re, sys
sys.stdout.reconfigure(encoding='utf-8')

logpath = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\Latest_trainingLogs\download (3).txt'
lines = open(logpath, encoding='utf-8', errors='replace').readlines()

pattern = re.compile(r'Epoch \d|BEST|improvement|checkpoint|NaN|Inf|recovery|Warmup|RESUME|CATASTROPHIC|cascade|DISABLED|Grad  \||Train \||Val   \||Ironclad|GRAD_CLIP|event')

for i, l in enumerate(lines):
    if pattern.search(l):
        print(f'{i+1}: {l.strip()}')
