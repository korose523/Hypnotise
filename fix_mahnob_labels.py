"""Fix MAHNOB labels using real self-assessment data from session.xml feltArsl (1-9).
Maps arousal 1-9 to 3-class: 1-3=Deep, 4-6=Light, 7-9=Awake."""
import os, json, numpy as np, re, sys
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load self-assessment from session.xml
with open('data/MAHNOB/mahnob_self_assessment.json', 'r', encoding='utf-8') as f:
    labels_527 = json.load(f)
sess2arousal = {str(l['session_id']): l['felt_arsl'] for l in labels_527}
print('Labeled sessions:', len(sess2arousal))

# Map arousal 1-9 to 3-class
def arousal_to_3class(val):
    if val <= 3:
        return 2  # Deep
    elif val <= 6:
        return 1  # Light
    else:
        return 0  # Awake

# Distribution of all 527 labels
all_labels_3c = [arousal_to_3class(v) for v in sess2arousal.values()]
dist = Counter(all_labels_3c)
print('3-class dist (all 527):', dict(dist))
print('Awake(0)=%d, Light(1)=%d, Deep(2)=%d' % (dist[0], dist[1], dist[2]))

# Now: map to prep01 windows
mahnob_npz = np.load('processed/prep01_windows/MAHNOB_windows.npz', allow_pickle=True)
subj_ids = mahnob_npz['subject_ids']
trial_ids = mahnob_npz['trial_ids']
n_wins = len(subj_ids)
print('\nMAHNOB windows:', n_wins)
print('Sample subject_ids:', subj_ids[:5])
print('Sample trial_ids:', trial_ids[:5])

# Build session_id -> window indices
session_windows = {}
for i in range(n_wins):
    subj = str(subj_ids[i])
    trial = str(trial_ids[i])
    sid_match = re.search(r'[Ss]ession[_\s]*(\d+)', subj)
    if not sid_match:
        sid_match = re.search(r'[Ss]ession[_\s]*(\d+)', trial)
    if not sid_match:
        sid_match = re.search(r'MAHNOB_(\d+)', subj)
    if not sid_match:
        sid_match = re.search(r'MAHNOB_(\d+)', trial)
    if sid_match:
        sess_id = sid_match.group(1)
        session_windows.setdefault(sess_id, []).append(i)

print('Unique sessions from prep01:', len(session_windows))
for sid, indices in list(session_windows.items())[:5]:
    print('  Session %s: %d windows' % (sid, len(indices)))

# Assign labels
win_labels = np.full(n_wins, -1, dtype=np.int32)
mapped, unmapped = 0, 0
for sess_id, indices in session_windows.items():
    if sess_id in sess2arousal:
        label_3c = arousal_to_3class(sess2arousal[sess_id])
        for idx in indices:
            win_labels[idx] = label_3c
        mapped += 1
    else:
        unmapped += 1

valid = win_labels >= 0
wdist = Counter(int(x) for x in win_labels[valid])
print('\nFinal: %d/%d windows labeled (%d sessions mapped, %d unmapped)'
      % (int(valid.sum()), n_wins, mapped, unmapped))
print('Label dist: Awake(0)=%d, Light(1)=%d, Deep(2)=%d, Unmapped(-1)=%d'
      % (wdist.get(0, 0), wdist.get(1, 0), wdist.get(2, 0),
         int((win_labels == -1).sum())))

if valid.sum() > 0:
    np.savez_compressed('processed/prep03_labels/MAHNOB_labels.npz',
                        labels=win_labels.astype(np.int32),
                        subject_ids=subj_ids)
    print('\nSaved MAHNOB_labels.npz')
    print('MAHNOB REAL AROUSAL LABELS FIXED!')
else:
    print('\nERROR: No valid labels!')
    sys.exit(1)

mahnob_npz.close()
