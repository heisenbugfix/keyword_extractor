import re

chunk = "fao animal"
term = "?fao"
if re.search(r'\b' + '('+term+')' + r'\b', chunk, re.IGNORECASE):
    print(True)
