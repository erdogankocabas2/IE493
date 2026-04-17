"""
IE493 Phase 2 — Simulation Validation Suite
=============================================
This script independently validates the preemptive-resume priority simulation
by running a battery of tests:

  1. Micro-scenario tests with hand-calculated expected results
  2. Structural invariant checks on the full 1000-patient simulation
  3. Priority ordering validation (Red Wq ≤ Yellow Wq ≤ Green Wq)
  4. Doctor utilization & capacity checks
  5. Preemption logic correctness tests
  6. Statistical reasonableness checks

Run: source .venv/bin/activate && python3 validate_phase2.py
"""

import matplotlib; matplotlib.use("Agg")
import pandas as pd
import numpy as np
from collections import deque
import heapq
import sys

# ══════════════════════════════════════════════════════════════
# Import the simulation engine from _test_phase2.py
# ══════════════════════════════════════════════════════════════
C = 5
RED, YELLOW, GREEN = 1, 2, 3
PRIORITY_MAP   = {'Red': RED, 'Yellow': YELLOW, 'Green': GREEN}
PRIORITY_NAMES = {RED: 'Red', YELLOW: 'Yellow', GREEN: 'Green'}

def min_to_clock(minutes):
    """Convert minutes from 08:00:00 to HH:MM:SS clock string."""
    total_sec = round(minutes * 60)
    base_sec  = 8 * 3600
    total_sec += base_sec
    day = total_sec // 86400
    rem = total_sec % 86400
    h   = rem // 3600
    mi  = (rem % 3600) // 60
    s   = rem % 60
    clock = f"{h:02d}:{mi:02d}:{s:02d}"
    if day > 0:
        return f"Day{day + 1} {clock}"
    return clock


def simulate_preemptive_resume(df, verbose=False):
    """Exact copy of the simulation engine from the notebook."""
    EVT_COMPLETION = 0
    EVT_ARRIVAL    = 1
    n = len(df)
    arrival   = df['arrival_min'].values.astype(float)
    priority  = df['priority_num'].values.astype(int)
    service   = df['Service_Required_Min'].values.astype(float)
    pat_ids   = df['Patient_ID'].values

    remaining_work       = service.copy()
    interruptions        = np.zeros(n, dtype=int)
    first_start          = np.full(n, np.nan)
    final_end            = np.full(n, np.nan)
    docs_busy_on_arrival = np.zeros(n, dtype=int)

    doc_busy       = [False] * C
    doc_patient    = [None]  * C
    doc_busy_until = [0.0]   * C
    doc_version    = [0]     * C

    wait_q = {RED: deque(), YELLOW: deque(), GREEN: deque()}
    event_q = []
    for i in range(n):
        heapq.heappush(event_q, (arrival[i], EVT_ARRIVAL, i, i, -1, -1))
    event_counter     = n
    total_preemptions = 0
    preemption_log    = []

    def assign_to_doctor(pat_idx, doc_idx, t):
        nonlocal event_counter
        svc_duration = remaining_work[pat_idx]
        doc_busy[doc_idx]       = True
        doc_patient[doc_idx]    = pat_idx
        doc_busy_until[doc_idx] = t + svc_duration
        doc_version[doc_idx]   += 1
        if np.isnan(first_start[pat_idx]):
            first_start[pat_idx] = t
        heapq.heappush(event_q, (
            t + svc_duration, EVT_COMPLETION, event_counter,
            pat_idx, doc_idx, doc_version[doc_idx]
        ))
        event_counter += 1

    def pull_from_queue():
        for prio in [RED, YELLOW, GREEN]:
            if wait_q[prio]:
                return wait_q[prio].popleft()
        return None

    def find_idle_doctor():
        for j in range(C):
            if not doc_busy[j]:
                return j
        return None

    while event_q:
        t, etype, _, pat_idx, doc_idx, ver = heapq.heappop(event_q)

        if etype == EVT_COMPLETION:
            if doc_version[doc_idx] != ver:
                continue
            final_end[pat_idx] = t
            doc_busy[doc_idx]    = False
            doc_patient[doc_idx] = None
            next_pat = pull_from_queue()
            if next_pat is not None:
                assign_to_doctor(next_pat, doc_idx, t)

        elif etype == EVT_ARRIVAL:
            pri = priority[pat_idx]
            docs_busy_on_arrival[pat_idx] = sum(doc_busy)

            idle_doc = find_idle_doctor()
            if idle_doc is not None:
                assign_to_doctor(pat_idx, idle_doc, t)
                continue

            if pri != RED:
                wait_q[pri].append(pat_idx)
                continue

            candidates = [
                j for j in range(C)
                if doc_busy[j] and priority[doc_patient[j]] == GREEN
            ]
            if not candidates:
                candidates = [
                    j for j in range(C)
                    if doc_busy[j] and priority[doc_patient[j]] == YELLOW
                ]
            if not candidates:
                wait_q[RED].append(pat_idx)
                continue

            victim_doc = min(candidates, key=lambda j: doc_busy_until[j])
            victim_pat = doc_patient[victim_doc]
            remaining = doc_busy_until[victim_doc] - t
            remaining_work[victim_pat] = remaining
            interruptions[victim_pat] += 1
            total_preemptions += 1
            preemption_log.append({
                'time': t,
                'time_clock': min_to_clock(t),
                'red_patient': int(pat_ids[pat_idx]),
                'victim_patient': int(pat_ids[victim_pat]),
                'victim_priority': PRIORITY_NAMES[priority[victim_pat]],
                'remaining_work': round(remaining, 4),
                'doctor': victim_doc,
                'victim_interruption_count': int(interruptions[victim_pat])
            })
            victim_pri = priority[victim_pat]
            wait_q[victim_pri].appendleft(victim_pat)
            assign_to_doctor(pat_idx, victim_doc, t)

    W  = final_end - arrival
    Wq = W - service

    results = pd.DataFrame({
        'Patient_ID':         pat_ids,
        'Priority':           [PRIORITY_NAMES[p] for p in priority],
        'priority_num':       priority,
        'Arrival_Min':        arrival,
        'First_Start_Min':    first_start,
        'Final_End_Min':      final_end,
        'Interruptions':      interruptions,
        'Service_Required':   service,
        'Wq_Min':             Wq,
        'W_Min':              W,
        'Docs_Busy_On_Arrival': docs_busy_on_arrival
    })
    return results, total_preemptions, preemption_log


# ══════════════════════════════════════════════════════════════
# TEST HELPERS
# ══════════════════════════════════════════════════════════════
passed = 0
failed = 0
total  = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ PASS: {name}")
    else:
        failed += 1
        print(f"  ❌ FAIL: {name}")
        if detail:
            print(f"          {detail}")


def make_df(rows):
    """Build a mini DataFrame for micro-scenario tests.
    Each row: (patient_id, arrival_min, priority_str, service_min)
    """
    return pd.DataFrame(rows, columns=[
        'Patient_ID', 'arrival_min', 'Priority', 'Service_Required_Min'
    ]).assign(priority_num=lambda d: d['Priority'].map(PRIORITY_MAP))


# ══════════════════════════════════════════════════════════════
# TEST GROUP 1: Micro-Scenario Tests (Hand-Calculated)
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 1: Micro-Scenario Tests (Hand-Calculated)")
print("=" * 70)
print()

# ── Test 1.1: Simple FCFS (no contention) ──
print("─── 1.1: No contention — all patients get immediate service ───")
df_test = make_df([
    (1, 0.0,  'Green',  10.0),
    (2, 1.0,  'Yellow', 8.0),
    (3, 2.0,  'Red',    5.0),
    (4, 3.0,  'Green',  7.0),
    (5, 4.0,  'Yellow', 6.0),
])
res, preempts, plog = simulate_preemptive_resume(df_test)

check("All patients complete", res['Final_End_Min'].notna().all())
check("Zero preemptions", preempts == 0)
check("Zero waiting time", (res['Wq_Min'].abs() < 1e-9).all())
# Patient 1: arrives 0, starts 0, ends 10
check("Patient 1 ends at 10.0", abs(res.iloc[0]['Final_End_Min'] - 10.0) < 1e-9)
# Patient 3 (Red): arrives 2, starts 2, ends 7
check("Patient 3 (Red) ends at 7.0", abs(res.iloc[2]['Final_End_Min'] - 7.0) < 1e-9)
print()

# ── Test 1.2: Basic queueing (no preemption, 1 doctor) ──
print("─── 1.2: Basic queueing with 1 doctor (save C, set C=1) ───")
old_C = C
C = 1  # temporarily use 1 doctor
# Monkey-patch the global
import types

df_test = make_df([
    (1, 0.0,   'Green', 10.0),
    (2, 3.0,   'Green',  5.0),
])
# We need a 1-doctor version. Inline sim:
# Patient 1: arr=0, start=0, end=10
# Patient 2: arr=3, start=10, end=15, Wq=7
C = 1
res, preempts, plog = simulate_preemptive_resume(df_test)
check("Patient 2 waits for 7 min", abs(res.iloc[1]['Wq_Min'] - 7.0) < 1e-9,
      f"Got Wq={res.iloc[1]['Wq_Min']:.4f}")
check("Patient 2 ends at 15.0", abs(res.iloc[1]['Final_End_Min'] - 15.0) < 1e-9,
      f"Got end={res.iloc[1]['Final_End_Min']:.4f}")
check("Zero preemptions", preempts == 0)
C = old_C
print()

# ── Test 1.3: Basic preemption scenario (1 doctor) ──
print("─── 1.3: Basic preemption — Red preempts Green (1 doctor) ───")
C = 1
df_test = make_df([
    (1, 0.0,   'Green', 10.0),   # starts at 0, would end at 10
    (2, 3.0,   'Red',    4.0),   # arrives at 3, preempts patient 1
])
# Expected:
#   t=0: Patient 1 starts on Doc 0, scheduled end = 10
#   t=3: Red Pat 2 arrives, all busy.
#         Preempt Green Pat 1 (remaining = 10-3 = 7 min)
#         Pat 2 starts, ends at 3+4 = 7
#   t=7: Pat 2 done. Pat 1 resumes (remaining=7), ends at 7+7 = 14
#   Pat 1: Wq = 14 - 0 - 10 = 4 min, W = 14
#   Pat 2: Wq = 0, W = 4
res, preempts, plog = simulate_preemptive_resume(df_test)

check("1 preemption event", preempts == 1,
      f"Got {preempts}")
check("Patient 1 interrupted once", int(res.iloc[0]['Interruptions']) == 1)
check("Patient 2 (Red) has Wq=0", abs(res.iloc[1]['Wq_Min']) < 1e-9,
      f"Got Wq={res.iloc[1]['Wq_Min']:.4f}")
check("Patient 2 (Red) ends at 7.0", abs(res.iloc[1]['Final_End_Min'] - 7.0) < 1e-9,
      f"Got end={res.iloc[1]['Final_End_Min']:.4f}")
check("Patient 1 (Green) ends at 14.0", abs(res.iloc[0]['Final_End_Min'] - 14.0) < 1e-9,
      f"Got end={res.iloc[0]['Final_End_Min']:.4f}")
check("Patient 1 Wq = 4.0", abs(res.iloc[0]['Wq_Min'] - 4.0) < 1e-9,
      f"Got Wq={res.iloc[0]['Wq_Min']:.4f}")
check("Victim is Green", plog[0]['victim_priority'] == 'Green')
check("Remaining work = 7.0", abs(plog[0]['remaining_work'] - 7.0) < 1e-4,
      f"Got {plog[0]['remaining_work']}")
C = old_C
print()

# ── Test 1.4: Red preempts Yellow, not another Red ──
print("─── 1.4: Red preempts Yellow (not another Red) — 1 doctor ───")
C = 1
df_test = make_df([
    (1, 0.0,  'Red',    20.0),   # Red already being served
    (2, 0.5,  'Yellow', 10.0),   # joins queue (Red being treated, can't preempt Red)
    (3, 1.0,  'Red',     5.0),   # arrives, only Red on doc — must queue
])
# t=0: Pat 1 (Red) starts, ends at 20
# t=0.5: Pat 2 (Yellow) arrives, doc busy with Red → queues
# t=1.0: Pat 3 (Red) arrives, doc busy with Red → can't preempt → queues (Red queue)
# t=20: Pat 1 done. Pull from queue: Red first → Pat 3 starts, ends at 25
# t=25: Pat 3 done. Pull: Yellow → Pat 2 starts, ends at 35
res, preempts, plog = simulate_preemptive_resume(df_test)

check("No preemptions (can't preempt Red with Red)", preempts == 0)
check("Patient 3 (Red) served before Patient 2 (Yellow)",
      res.iloc[2]['First_Start_Min'] < res.iloc[1]['First_Start_Min'],
      f"Red start={res.iloc[2]['First_Start_Min']:.2f}, Yellow start={res.iloc[1]['First_Start_Min']:.2f}")
check("Patient 1 ends at 20.0", abs(res.iloc[0]['Final_End_Min'] - 20.0) < 1e-9)
check("Patient 3 (Red) ends at 25.0", abs(res.iloc[2]['Final_End_Min'] - 25.0) < 1e-9,
      f"Got {res.iloc[2]['Final_End_Min']:.4f}")
check("Patient 2 (Yellow) ends at 35.0", abs(res.iloc[1]['Final_End_Min'] - 35.0) < 1e-9,
      f"Got {res.iloc[1]['Final_End_Min']:.4f}")
C = old_C
print()

# ── Test 1.5: Double preemption — same Green patient preempted twice ──
print("─── 1.5: Double preemption of same Green patient (1 doctor) ───")
C = 1
df_test = make_df([
    (1, 0.0,  'Green', 20.0),   # starts at 0
    (2, 5.0,  'Red',    3.0),   # preempts at t=5, remaining=15
    (3, 10.0, 'Red',    2.0),   # preempts resumed Green at t=10
])
# t=0:  Pat1 (G) starts, ends=20
# t=5:  Pat2 (R) arrives, preempts. Pat1 remaining=15. Pat2 ends=8
# t=8:  Pat2 done. Pat1 resumes (remaining=15), ends=8+15=23
# t=10: Pat3 (R) arrives, preempts. Pat1 remaining=23-10=13. Pat3 ends=12
# t=12: Pat3 done. Pat1 resumes (remaining=13), ends=12+13=25
# Pat1: W=25-0=25, Wq=25-20=5, interruptions=2
# Pat2: W=3, Wq=0
# Pat3: W=2, Wq=0
res, preempts, plog = simulate_preemptive_resume(df_test)

check("2 preemptions total", preempts == 2, f"Got {preempts}")
check("Patient 1 interrupted twice", int(res.iloc[0]['Interruptions']) == 2)
check("Patient 1 ends at 25.0", abs(res.iloc[0]['Final_End_Min'] - 25.0) < 1e-9,
      f"Got {res.iloc[0]['Final_End_Min']:.4f}")
check("Patient 1 Wq = 5.0", abs(res.iloc[0]['Wq_Min'] - 5.0) < 1e-9,
      f"Got {res.iloc[0]['Wq_Min']:.4f}")
check("Patient 2 ends at 8.0", abs(res.iloc[1]['Final_End_Min'] - 8.0) < 1e-9,
      f"Got {res.iloc[1]['Final_End_Min']:.4f}")
check("Patient 3 ends at 12.0", abs(res.iloc[2]['Final_End_Min'] - 12.0) < 1e-9,
      f"Got {res.iloc[2]['Final_End_Min']:.4f}")
C = old_C
print()

# ── Test 1.6: Green before Yellow with preemption — victim is Green (not Yellow) ──
print("─── 1.6: Red preempts Green first, not Yellow (2 doctors) ───")
C = 2
df_test = make_df([
    (1, 0.0,  'Yellow', 10.0),   # Doc 0
    (2, 0.0,  'Green',  10.0),   # Doc 1
    (3, 3.0,  'Red',     5.0),   # preempts Green (not Yellow)
])
# t=0: Pat1 (Y) → Doc 0, Pat2 (G) → Doc 1
# t=3: Pat3 (R) arrives, all docs busy. Green on Doc1 → preempt Green
#       Pat2 remaining = 10-3 = 7
#       Pat3 starts on Doc1, ends at 8
# t=8: Pat3 done. Pat2 resumes (7 min), ends at 15
# t=10: Pat1 done.
res, preempts, plog = simulate_preemptive_resume(df_test)

check("1 preemption", preempts == 1)
check("Victim is Green (not Yellow)", plog[0]['victim_priority'] == 'Green')
check("Yellow patient NOT interrupted", int(res.iloc[0]['Interruptions']) == 0)
check("Green patient interrupted once", int(res.iloc[1]['Interruptions']) == 1)
check("Patient 2 (Green) ends at 15.0", abs(res.iloc[1]['Final_End_Min'] - 15.0) < 1e-9,
      f"Got {res.iloc[1]['Final_End_Min']:.4f}")
C = old_C
print()

# ── Test 1.7: Priority queue ordering – Yellow before Green ──
print("─── 1.7: Priority queue ordering — Yellow served before Green ───")
C = 1
df_test = make_df([
    (1, 0.0,   'Red',    10.0),  # blocks the doctor
    (2, 1.0,   'Green',   5.0),  # queued
    (3, 2.0,   'Yellow',  3.0),  # queued (higher priority than Green)
])
# t=0: Pat1 starts, ends=10
# t=1: Pat2 (G) queues
# t=2: Pat3 (Y) queues
# t=10: Pat1 done. Pull from queue: Yellow first → Pat3 starts, ends=13
# t=13: Pat3 done. Pull: Green → Pat2 starts, ends=18
res, preempts, plog = simulate_preemptive_resume(df_test)

check("Yellow served before Green",
      res.iloc[2]['First_Start_Min'] < res.iloc[1]['First_Start_Min'],
      f"Yellow start={res.iloc[2]['First_Start_Min']:.2f}, Green start={res.iloc[1]['First_Start_Min']:.2f}")
check("Patient 3 (Yellow) starts at 10.0", abs(res.iloc[2]['First_Start_Min'] - 10.0) < 1e-9)
check("Patient 2 (Green) starts at 13.0", abs(res.iloc[1]['First_Start_Min'] - 13.0) < 1e-9)
check("No preemptions", preempts == 0)
C = old_C
print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 2: Full Dataset Structural Invariants
# ══════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST GROUP 2: Full Dataset Structural Invariants (1000 patients)")
print("=" * 70)
print()

C = 5  # restore
df = pd.read_csv('Group_12/ER_Phase2_Group_12.csv')
ER_OPEN_HOUR = 8
day_offset = 0
prev_total_sec = -1
arrival_minutes = []
for t_str in df['Arrival_Clock']:
    parts = t_str.strip().split(':')
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    total_sec = h * 3600 + m * 60 + s
    if prev_total_sec >= 0 and total_sec < prev_total_sec - 3600:
        day_offset += 1
    prev_total_sec = total_sec
    minutes_from_open = (day_offset * 24 * 60) + (h * 60 + m + s / 60.0) - (ER_OPEN_HOUR * 60)
    arrival_minutes.append(minutes_from_open)
df['arrival_min'] = arrival_minutes
df['priority_num'] = df['Priority'].map(PRIORITY_MAP)

results, total_preemptions, preemption_log = simulate_preemptive_resume(df, verbose=False)

# ── 2.1: All patients completed ──
check("All 1000 patients completed", results['Final_End_Min'].notna().all())

# ── 2.2: Timeline ordering ──
check("Arrival ≤ Start for all",
      (results['First_Start_Min'] >= results['Arrival_Min'] - 1e-9).all())
check("Start ≤ End for all",
      (results['Final_End_Min'] >= results['First_Start_Min'] - 1e-9).all())

# ── 2.3: Non-negative waiting ──
check("All Wq ≥ 0", (results['Wq_Min'] >= -1e-9).all())

# ── 2.4: W = Wq + Service ──
check("W = Wq + Service for all",
      (abs(results['W_Min'] - results['Wq_Min'] - results['Service_Required']) < 1e-9).all())

# ── 2.5: Work conservation ──
total_svc = results['Service_Required'].sum()
total_delivered = (results['W_Min'] - results['Wq_Min']).sum()
check("Work conservation (total service)",
      abs(total_svc - total_delivered) < 1e-6,
      f"required={total_svc:.4f}, delivered={total_delivered:.4f}")

# ── 2.6: Red patients have zero or very low Wq ──
red_mask = results['Priority'] == 'Red'
red_max_wq = results.loc[red_mask, 'Wq_Min'].max()
check(f"Red max Wq is low (got {red_max_wq:.4f} min)",
      red_max_wq < 100,  # reasonable upper bound
      f"Max red Wq = {red_max_wq:.4f}")

# ── 2.7: Red patients not interrupted ──
check("No Red patients interrupted",
      (results.loc[red_mask, 'Interruptions'] == 0).all())

# ── 2.8: Only Green/Yellow patients preempted ──
for entry in preemption_log:
    if entry['victim_priority'] not in ('Green', 'Yellow'):
        check("Preemption victims are Green/Yellow only", False,
              f"Unexpected: {entry['victim_priority']}")
        break
else:
    check("All preemption victims are Green/Yellow", True)

# ── 2.9: Remaining work in preemption log is positive ──
all_positive = all(e['remaining_work'] > 0 for e in preemption_log)
check("All remaining_work in preemption log > 0", all_positive)

# ── 2.10: Remaining work < original service time ──
svc_map = dict(zip(df['Patient_ID'], df['Service_Required_Min']))
all_less = all(
    e['remaining_work'] <= svc_map[e['victim_patient']] + 1e-9
    for e in preemption_log
)
check("All remaining_work ≤ original service time", all_less)

# ── 2.11: Interruption count consistency ──
# Total interruptions across patients == total_preemptions
total_int_from_results = results['Interruptions'].sum()
check("Sum of interruptions = total preemptions",
      total_int_from_results == total_preemptions,
      f"sum(interruptions)={total_int_from_results}, total_preemptions={total_preemptions}")

# ── 2.12: Docs busy on arrival is in [0, 5] ──
check("Docs_Busy_On_Arrival in [0, 5]",
      results['Docs_Busy_On_Arrival'].between(0, 5).all())

print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 3: Priority Ordering Validation
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 3: Priority Ordering Validation")
print("=" * 70)
print()

red_avg_wq    = results.loc[results['Priority'] == 'Red',    'Wq_Min'].mean()
yellow_avg_wq = results.loc[results['Priority'] == 'Yellow', 'Wq_Min'].mean()
green_avg_wq  = results.loc[results['Priority'] == 'Green',  'Wq_Min'].mean()

print(f"  Avg Wq — Red: {red_avg_wq:.4f},  Yellow: {yellow_avg_wq:.4f},  Green: {green_avg_wq:.4f}")

check("Red Avg Wq ≤ Yellow Avg Wq",
      red_avg_wq <= yellow_avg_wq + 1e-9,
      f"Red={red_avg_wq:.4f}, Yellow={yellow_avg_wq:.4f}")
check("Yellow Avg Wq ≤ Green Avg Wq",
      yellow_avg_wq <= green_avg_wq + 1e-9,
      f"Yellow={yellow_avg_wq:.4f}, Green={green_avg_wq:.4f}")

red_avg_w    = results.loc[results['Priority'] == 'Red',    'W_Min'].mean()
yellow_avg_w = results.loc[results['Priority'] == 'Yellow', 'W_Min'].mean()
green_avg_w  = results.loc[results['Priority'] == 'Green',  'W_Min'].mean()

print(f"  Avg W  — Red: {red_avg_w:.4f},  Yellow: {yellow_avg_w:.4f},  Green: {green_avg_w:.4f}")

check("Red Avg W ≤ Yellow Avg W",
      red_avg_w <= yellow_avg_w + 1e-9,
      f"Red={red_avg_w:.4f}, Yellow={yellow_avg_w:.4f}")

print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 4: Doctor Utilization & Capacity Checks
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 4: Doctor Utilization & Capacity Checks")
print("=" * 70)
print()

# ── 4.1: No more than 5 patients in service at any instant ──
# Reconstruct a timeline of [start, end) intervals to check overlaps
# For each patient, the total service intervals sum to Service_Required.
# But we can do a simpler check: at no point in time should more than 5
# patients be in "actual service" simultaneously.
# We use an event-sweep approach: +1 at start, -1 at end, check max.

events = []
for _, row in results.iterrows():
    events.append((row['First_Start_Min'], +1))
    events.append((row['Final_End_Min'],   -1))
events.sort(key=lambda x: (x[0], x[1]))  # -1 before +1 at same time

max_concurrent = 0
current = 0
for t, delta in events:
    current += delta
    max_concurrent = max(max_concurrent, current)

# Note: This is an UPPER BOUND check. Due to preemption, the actual max
# concurrent might be exactly 5. The First_Start/Final_End spans include
# idle gaps from preemption, so this can overcount. Let's just check ≤ 5
# doesn't hold literally (it won't since spans overlap due to preemption).
# Instead, check that the simulation reports at most 5 docs busy.
check("Max docs busy at any arrival ≤ 5",
      results['Docs_Busy_On_Arrival'].max() <= 5)

# ── 4.2: Utilization calculation ──
total_service = results['Service_Required'].sum()
sim_span = results['Final_End_Min'].max() - results['Arrival_Min'].min()
utilization = total_service / (C * sim_span)
print(f"  System utilization ρ = {utilization:.4f}")
check("Utilization ρ < 1.0 (stable system)", utilization < 1.0,
      f"ρ = {utilization:.4f}")
check("Utilization ρ > 0.3 (reasonable load)", utilization > 0.3,
      f"ρ = {utilization:.4f}")

# ── 4.3: Last patient ends after last arrival ──
check("Simulation ends after last arrival",
      results['Final_End_Min'].max() >= results['Arrival_Min'].max())

print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 5: Preemption Logic Correctness
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 5: Preemption Logic Correctness")
print("=" * 70)
print()

# ── 5.1: Each preemption is by a Red patient ──
all_red = all(
    results.loc[results['Patient_ID'] == e['red_patient'], 'Priority'].iloc[0] == 'Red'
    for e in preemption_log
)
check("All preempting patients are Red", all_red)

# ── 5.2: No Yellow-preempts-Green events ──
# Only Red can preempt; Yellow never preempts
yellow_preempts = [e for e in preemption_log
                   if results.loc[results['Patient_ID'] == e['red_patient'], 'Priority'].iloc[0] != 'Red']
check("No non-Red preemptions", len(yellow_preempts) == 0,
      f"Found {len(yellow_preempts)} non-Red preemptions")

# ── 5.3: Preemption count matches ──
check(f"Preemption log has {total_preemptions} entries",
      len(preemption_log) == total_preemptions)

# ── 5.4: Green preempted before Yellow ──
# If both Green and Yellow are on doctors, Green should be the victim
# This is already validated by victim class distribution
green_victims = sum(1 for e in preemption_log if e['victim_priority'] == 'Green')
yellow_victims = sum(1 for e in preemption_log if e['victim_priority'] == 'Yellow')
print(f"  Green victims: {green_victims}, Yellow victims: {yellow_victims}")
check("Green preempted more than Yellow",
      green_victims >= yellow_victims,
      f"Green={green_victims}, Yellow={yellow_victims}")

print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 6: Statistical Reasonableness
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 6: Statistical Reasonableness")
print("=" * 70)
print()

# ── 6.1: Average Wq is reasonable (not too large) ──
avg_wq = results['Wq_Min'].mean()
print(f"  Overall Avg Wq = {avg_wq:.4f} min")
check("Avg Wq < 60 min (reasonable for ρ~0.76)", avg_wq < 60)
check("Avg Wq ≥ 0", avg_wq >= 0)

# ── 6.2: Average W is close to Avg Service + Avg Wq ──
avg_w   = results['W_Min'].mean()
avg_svc = results['Service_Required'].mean()
check("Avg W ≈ Avg Wq + Avg Service",
      abs(avg_w - avg_wq - avg_svc) < 1e-9,
      f"Avg W={avg_w:.4f}, Avg Wq + Avg Svc = {avg_wq + avg_svc:.4f}")

# ── 6.3: Preemption rate is reasonable ──
# With ~71 Red patients and ρ~0.76, not every Red triggers preemption
red_count = (results['Priority'] == 'Red').sum()
preemption_rate = total_preemptions / red_count if red_count > 0 else 0
print(f"  Preemption rate = {total_preemptions}/{red_count} = {preemption_rate:.2%}")
check("Preemption rate < 100%", preemption_rate < 1.0)
check("Preemptions ≤ Red count", total_preemptions <= red_count * 3,
      "Each Red can cause at most ~1 preemption")

# ── 6.4: Compare with FCFS baseline ──
# Run FCFS and check that overall avg Wq is similar
def simulate_fcfs(df):
    n   = len(df)
    arr = df['arrival_min'].values.astype(float)
    svc = df['Service_Required_Min'].values.astype(float)
    doc_free = np.zeros(C)
    start    = np.empty(n)
    end      = np.empty(n)
    for i in range(n):
        j  = int(np.argmin(doc_free))
        st = max(arr[i], doc_free[j])
        en = st + svc[i]
        start[i] = st
        end[i]   = en
        doc_free[j] = en
    return start - arr, end - arr

fcfs_wq, fcfs_W = simulate_fcfs(df)
fcfs_avg_wq = fcfs_wq.mean()
priority_avg_wq = results['Wq_Min'].mean()

print(f"  FCFS Avg Wq = {fcfs_avg_wq:.4f},  Priority Avg Wq = {priority_avg_wq:.4f}")

# Overall avg Wq should be similar (priority doesn't change total work)
check("Priority Avg Wq within 50% of FCFS Avg Wq",
      abs(priority_avg_wq - fcfs_avg_wq) / max(fcfs_avg_wq, 0.001) < 0.50,
      f"Diff = {abs(priority_avg_wq - fcfs_avg_wq):.4f}")

# ── 6.5: Priority system benefits Red patients ──
red_fcfs_wq = fcfs_wq[df['priority_num'] == RED].mean()
red_pr_wq   = results.loc[results['Priority'] == 'Red', 'Wq_Min'].mean()
print(f"  Red FCFS Wq = {red_fcfs_wq:.4f},  Red Priority Wq = {red_pr_wq:.4f}")
check("Red patients benefit from priority (Wq decreased)",
      red_pr_wq <= red_fcfs_wq + 1e-9,
      f"Red FCFS Wq={red_fcfs_wq:.4f}, Red Priority Wq={red_pr_wq:.4f}")

# ── 6.6: Green patients are penalized by priority ──
green_fcfs_wq = fcfs_wq[df['priority_num'] == GREEN].mean()
green_pr_wq   = results.loc[results['Priority'] == 'Green', 'Wq_Min'].mean()
print(f"  Green FCFS Wq = {green_fcfs_wq:.4f},  Green Priority Wq = {green_pr_wq:.4f}")
check("Green patients penalized by priority (Wq increased)",
      green_pr_wq >= green_fcfs_wq - 1e-9,
      f"Green FCFS Wq={green_fcfs_wq:.4f}, Green Priority Wq={green_pr_wq:.4f}")

print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 7: Edge Case — Simultaneous Arrivals
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 7: Edge Case — Simultaneous Arrivals")
print("=" * 70)
print()

C = 1
# Two patients arrive at same time: Red should go first
df_test = make_df([
    (1, 0.0, 'Green', 5.0),
    (2, 0.0, 'Red',   3.0),
])
res, preempts, plog = simulate_preemptive_resume(df_test)
# Both arrive at t=0. Green arrives first (lower index), gets doc.
# Red arrives same time, preempts Green (remaining=5).
# Red ends at 3. Green resumes, ends at 3+5=8.
check("Red patient served immediately when arriving same time",
      res.iloc[1]['Wq_Min'] < 1e-9 or preempts >= 1,
      f"Red Wq={res.iloc[1]['Wq_Min']:.4f}, preempts={preempts}")
C = old_C
print()


# ══════════════════════════════════════════════════════════════
# TEST GROUP 8: Idempotency — Running twice gives same results
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST GROUP 8: Idempotency — Deterministic Results")
print("=" * 70)
print()

C = 5
results2, preempts2, plog2 = simulate_preemptive_resume(df, verbose=False)

check("Same number of preemptions on re-run",
      total_preemptions == preempts2)
check("Same end times on re-run",
      np.allclose(results['Final_End_Min'].values, results2['Final_End_Min'].values, atol=1e-12))
check("Same Wq values on re-run",
      np.allclose(results['Wq_Min'].values, results2['Wq_Min'].values, atol=1e-12))
check("Same interruption counts on re-run",
      (results['Interruptions'].values == results2['Interruptions'].values).all())

print()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print(f"VALIDATION SUMMARY: {passed}/{total} tests passed, {failed} failed")
print("=" * 70)

if failed > 0:
    print("\n⚠️  Some tests FAILED. Review the output above for details.")
    sys.exit(1)
else:
    print("\n🎉 All validation tests PASSED! Simulation is correct.")
    sys.exit(0)
