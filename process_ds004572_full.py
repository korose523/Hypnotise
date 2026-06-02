"""Process ALL 52 subjects of ds004572 using lazy loading + resampling.
Run: NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp python process_ds004572_full.py
"""
import os, sys
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['MPLCONFIGDIR'] = os.environ.get('MPLCONFIGDIR', os.environ.get('TEMP', '/tmp'))
import mne, numpy as np, time, json, warnings
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\mac\WorkBuddy\2026-05-13-task-2\universal_bci_hypnosis')
sys.path.insert(0, '.')
from shared.feature_extraction import map_channels_to_14, EPOC_CHANNELS
from pathlib import Path

data_dir = Path('data/ds004572')
vhdr_files = sorted(data_dir.rglob('*.vhdr'))
subjs = sorted(set(f.parent.parent.parent.name for f in vhdr_files))
print(f'ds004572: {len(subjs)} subjects, {len(vhdr_files)} files')

out_dir = Path('processed/ds004572_checkpoint')
out_dir.mkdir(parents=True, exist_ok=True)

# Check for resume
all_windows = []; all_subjs = []; all_trials = []
completed_subjs = set()
ckpt_path = out_dir / 'checkpoint.npz'
progress_path = out_dir / 'progress.json'
if ckpt_path.exists():
    ckpt = np.load(ckpt_path, allow_pickle=True)
    all_windows = list(ckpt['windows'])
    all_subjs = list(ckpt['subjects'])
    all_trials = list(ckpt['trials'])
    if progress_path.exists():
        completed_subjs = set(json.loads(progress_path.read_text()).get('done', []))
    print(f'Resuming: {len(completed_subjs)} subjects done, {len(all_windows)} windows so far')

t0 = time.time()
for subj in subjs:
    if subj in completed_subjs:
        continue
    print(f'[{subj}] Processing...', flush=True)
    subj_files = [f for f in vhdr_files if subj in str(f)]
    subj_wins = 0
    for vf in subj_files:
        try:
            raw = mne.io.read_raw_brainvision(str(vf), preload=False, verbose='ERROR')
            raw.resample(128, verbose='ERROR')
            data, _ = raw[:, :]
            eeg_14ch, _ = map_channels_to_14(data, raw.ch_names, EPOC_CHANNELS)
            wl, sl = 256, 128
            nw = max(0, (eeg_14ch.shape[0] - wl) // sl + 1)
            for s in range(0, nw * sl, sl):
                all_windows.append(eeg_14ch[s:s+wl].astype(np.float32))
                all_subjs.append(f'ds004572_{subj}')
                all_trials.append(vf.stem)
            subj_wins += nw
        except Exception as e:
            print(f'  Skip {vf.name}: {e}')
    
    completed_subjs.add(subj)
    print(f'[{subj}] {subj_wins} windows (total: {len(all_windows)}, {len(completed_subjs)}/{len(subjs)} done, {time.time()-t0:.0f}s)', flush=True)
    
    # Save checkpoint every 5 subjects
    if len(completed_subjs) % 5 == 0:
        np.savez_compressed(ckpt_path, windows=np.array(all_windows, dtype=object),
                           subjects=np.array(all_subjs), trials=np.array(all_trials))
        progress_path.write_text(json.dumps({'done': list(completed_subjs)}))
        print(f'  Checkpoint saved', flush=True)

# Final save
elapsed = time.time() - t0
arr = np.stack(all_windows) if len(all_windows) > 0 else np.zeros((0, 256, 14))
print(f'\nTotal: {len(all_windows)} windows in {elapsed:.0f}s ({elapsed/3600:.1f}h)')

# Save prep01 output
prep01_dir = Path('processed/prep01_windows')
np.savez_compressed(prep01_dir / 'ds004572_windows.npz',
                    windows=arr, subject_ids=np.array(all_subjs), trial_ids=np.array(all_trials))

# Update summary
summary_path = prep01_dir / 'prep01_summary.json'
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary['ds004572'] = {'status': 'ok', 'n_windows': len(all_windows), 'window_shape': [256, 14],
                       'n_subjects': len(subjs), 'save_path': str(prep01_dir / 'ds004572_windows.npz')}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

# Clean checkpoint
import shutil; shutil.rmtree(out_dir)
print('Done! Checkpoint cleaned.')
