import os
import re
import requests
from dotenv import load_dotenv

# Script name: check_status.py

load_dotenv()
email = os.getenv('ZTF_EMAIL', 'medhansh29@gmail.com')
userpass = os.getenv('ZTF_USERPASS', 'garg015')

print("Querying ZTF Batch Forced Photometry database...\n")

# --- 1. Check Completed Jobs ---
settings_recent = {
    'email': email, 
    'userpass': userpass,
    'option': 'All recent jobs', 
    'action': 'Query Database'
}
r_recent = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi',
                        auth=('ztffps', 'dontgocrazy!'), params=settings_recent)

if r_recent.status_code == 200:
    wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! -O '
    wget_url = 'https://ztfweb.ipac.caltech.edu'
    wget_suffix = '"'
    
    lightcurves = re.findall(r'/ztf/ops.+?lc\.txt\b', r_recent.text)
    lightcurves += re.findall(r'/ztf/ops.+?\.tar\.gz\b', r_recent.text)
    
    if lightcurves:
        print(f"✅ Found {len(lightcurves)} completed jobs ready for download:")
        for lc in lightcurves:
            p = re.match(r'.+/(.+)', lc)
            if p:
                print(wget_prefix + p.group(1) + ' "' + wget_url + lc + wget_suffix)
    else:
        print("❌ No completed lightcurve links found.")
else:
    print(f"Error querying recent jobs: HTTP {r_recent.status_code}")

print("\n--------------------------------------------------\n")

# --- 2. Check Pending Jobs ---
settings_pending = {
    'email': email, 
    'userpass': userpass,
    'option': 'Pending jobs', 
    'action': 'Query Database'
}
r_pending = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi',
                         auth=('ztffps', 'dontgocrazy!'), params=settings_pending)

if r_pending.status_code == 200:
    # Extract "XXX records returned" from the HTML
    match = re.search(r'(\d+)\s+records returned', r_pending.text)
    if match:
        pending_count = int(match.group(1))
        if pending_count > 0:
            print(f"⏳ You currently have {pending_count} jobs QUEUED or PROCESSING on ZTF's servers.")
        else:
            print("✅ You have 0 pending jobs in the queue.")
    else:
        print("Could not parse pending jobs count, but query was successful.")
else:
    print(f"Error querying pending jobs: HTTP {r_pending.status_code}")