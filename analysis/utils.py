#!/usr/bin/env python3.8

# --- Positions ---
POSITIONS = ['sitting', 'standing']

# --- Setups and Legends ---
SETUPS = ['remote', 'head_mounted']
SETUPS_LEGEND = {
    'remote': 'Remote',
    'head_mounted': 'Head Mounted'
}

# --- Devices and Legends ---
DEVICES = ['fovio', 'pupil', 'tobii']
DEVICES_LEGEND = {
    'fovio': 'Remote (Fovio FX3)',
    'pupil': 'Head Mounted (Pupil Core)',
    'tobii': 'Remote (Tobii Pro 4C)'
}

# --- Figures ---
FIGURES = ['car', 'tb', 'house', 'sc', 'tc', 'tsb']

# --- Scores and Legends ---
SCORES = ['impossible', 'severe_issues', 'slight_issues', 'no_issue']
SCORES_LEGEND = {
    'impossible': "Impossible",
    'severe_issues': 'Severe\nIssues',
    'slight_issues': 'Slight\nIssues',
    'no_issue': 'No Issue'
}

# --- Scenes ---
SCENES = ['table', 'screen']

# --- Dataset Path ---
DATASET = '../gaipat'
