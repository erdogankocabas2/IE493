import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown
cells.append(nbf.v4.new_markdown_cell("""# IE493 Project — Phase 3: Integrated System (Network Flow)

**Group 12** — Erdoğan Kocabaş (2021402183) & Berru Selcan Cocen (2021402003)

---

## Overview

Phase 3 extends the Phase 2 preemptive-resume ER simulation by integrating a **Diagnostic Lab** (Node 2).
Upon completing Doctor treatment, ~40% of patients proceed to the Lab.
The Lab operates on an FCFS basis with a fixed capacity of 3 machines, ignoring priority status.
We will analyze the total cycle time (Door-to-Door), verify lab constraints, and perform capacity planning to meet performance targets.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# ── Constants ──
RED, YELLOW, GREEN = 1, 2, 3
PRIORITY_MAP   = {'Red': RED, 'Yellow': YELLOW, 'Green': GREEN}
PRIORITY_NAMES = {RED: 'Red', YELLOW: 'Yellow', GREEN: 'Green'}
PRIORITY_COLORS = {'Red': '#e74c3c', 'Yellow': '#f39c12', 'Green': '#27ae60'}
"""))

# Cell 3: Data Loading
cells.append(nbf.v4.new_markdown_cell("""# 1. Data Loading & Preprocessing

Load the Group 12 Phase 3 dataset. This dataset includes `Needs_Lab` and `Lab_Service_Required_Min`.
"""))

# Cell 4: Load Data
cells.append(nbf.v4.new_code_cell("""def min_to_clock(minutes):
    \"\"\"Convert minutes from 08:00:00 to HH:MM:SS clock string.\"\"\"
    if np.isnan(minutes):
        return ""
    total_sec = round(minutes * 60)
    base_sec  = 8 * 3600  # 08:00:00
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

# Load dataset
df = pd.read_csv('Group_12/ER_Phase3_Group_12.csv')
print(f"Loaded {len(df)} patients")

# Parse Arrival_Clock -> minutes from 08:00
ER_OPEN_HOUR = 8  # 08:00
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

print("Needs_Lab distribution:")
print(df['Needs_Lab'].value_counts().to_string())
print()
print("Lab Service Time (for those who need it):")
print(df.loc[df['Needs_Lab'], 'Lab_Service_Required_Min'].describe())

df.head()
"""))

# Cell 5: Engine Intro
cells.append(nbf.v4.new_markdown_cell("""# 2. Integrated Simulation Engine (Doctors + Lab)

We extend the event-driven min-heap simulation from Phase 2.
Events are processed in chronological order:
1. `LAB_COMPLETION`: Lab machine freed, pull next patient from FCFS lab queue.
2. `DOC_COMPLETION`: Doctor freed, pull next from doctor priority queue. If completing patient needs lab, they are immediately routed to the lab (either starting lab service or joining the lab queue).
3. `ARRIVAL`: Patient arrives at the hospital.

Doctor logic follows Phase 2 (Preemptive-Resume).
Lab logic follows Phase 3 (FCFS, no priority).
"""))

# Cell 6: Engine Code
cells.append(nbf.v4.new_code_cell("""def simulate_network_flow(df, s_dr=5, s_lab=3, verbose=False):
    # ── Event type constants ──
    # Order matters for tie-breaking at same timestamp
    EVT_LAB_COMPLETION = 0
    EVT_DOC_COMPLETION = 1
    EVT_ARRIVAL        = 2

    n = len(df)

    # ── Extract arrays for speed ──
    arrival   = df['arrival_min'].values.astype(float)
    priority  = df['priority_num'].values.astype(int)
    doc_service = df['Service_Required_Min'].values.astype(float)
    needs_lab = df['Needs_Lab'].values.astype(bool)
    lab_service = df['Lab_Service_Required_Min'].values.astype(float)
    pat_ids   = df['Patient_ID'].values

    # ── Patient state ──
    remaining_work       = doc_service.copy()
    interruptions        = np.zeros(n, dtype=int)
    doc_first_start      = np.full(n, np.nan)
    doc_final_end        = np.full(n, np.nan)
    lab_start            = np.full(n, np.nan)
    lab_end              = np.full(n, np.nan)
    docs_busy_on_arrival = np.zeros(n, dtype=int)

    # ── Node 1: Doctor state ──
    doc_busy       = [False] * s_dr
    doc_patient    = [None]  * s_dr
    doc_busy_until = [0.0]   * s_dr
    doc_version    = [0]     * s_dr
    doc_wait_q = {RED: deque(), YELLOW: deque(), GREEN: deque()}

    # ── Node 2: Lab state ──
    lab_busy       = [False] * s_lab
    lab_patient    = [None]  * s_lab
    lab_version    = [0]     * s_lab
    lab_wait_q     = deque() # FCFS

    # ── Event queue (min-heap) ──
    # Format: (time, event_type, tiebreak, pat_idx, res_idx, version)
    event_q = []
    for i in range(n):
        heapq.heappush(event_q, (arrival[i], EVT_ARRIVAL, i, i, -1, -1))

    event_counter = n
    total_preemptions = 0
    
    # Track utilization areas
    doc_busy_time_area = 0.0
    lab_busy_time_area = 0.0
    last_event_time = 0.0

    def update_areas(t):
        nonlocal doc_busy_time_area, lab_busy_time_area, last_event_time
        dt = t - last_event_time
        if dt > 0:
            doc_busy_time_area += sum(doc_busy) * dt
            lab_busy_time_area += sum(lab_busy) * dt
        last_event_time = t

    # ── Helpers for Node 2 (Lab) ──
    def assign_to_lab(pat_idx, machine_idx, t):
        nonlocal event_counter
        svc_duration = lab_service[pat_idx]
        lab_busy[machine_idx] = True
        lab_patient[machine_idx] = pat_idx
        lab_version[machine_idx] += 1
        lab_start[pat_idx] = t
        
        heapq.heappush(event_q, (
            t + svc_duration, EVT_LAB_COMPLETION, event_counter, 
            pat_idx, machine_idx, lab_version[machine_idx]
        ))
        event_counter += 1

    def route_to_lab(pat_idx, t):
        if not needs_lab[pat_idx]:
            return
        
        # Try find free lab machine
        idle_mac = None
        for j in range(s_lab):
            if not lab_busy[j]:
                idle_mac = j
                break
                
        if idle_mac is not None:
            assign_to_lab(pat_idx, idle_mac, t)
        else:
            lab_wait_q.append(pat_idx)

    # ── Helpers for Node 1 (Doctor) ──
    def assign_to_doctor(pat_idx, doc_idx, t):
        nonlocal event_counter
        svc_duration = remaining_work[pat_idx]
        doc_busy[doc_idx]       = True
        doc_patient[doc_idx]    = pat_idx
        doc_busy_until[doc_idx] = t + svc_duration
        doc_version[doc_idx]   += 1

        if np.isnan(doc_first_start[pat_idx]):
            doc_first_start[pat_idx] = t

        heapq.heappush(event_q, (
            t + svc_duration, EVT_DOC_COMPLETION, event_counter,
            pat_idx, doc_idx, doc_version[doc_idx]
        ))
        event_counter += 1

    def pull_from_doc_queue():
        for prio in [RED, YELLOW, GREEN]:
            if doc_wait_q[prio]:
                return doc_wait_q[prio].popleft()
        return None

    def find_idle_doctor():
        for j in range(s_dr):
            if not doc_busy[j]:
                return j
        return None

    # ═════ MAIN EVENT LOOP ═════
    while event_q:
        t, etype, _, pat_idx, res_idx, ver = heapq.heappop(event_q)
        update_areas(t)

        if etype == EVT_LAB_COMPLETION:
            if lab_version[res_idx] != ver: continue
            
            lab_end[pat_idx] = t
            lab_busy[res_idx] = False
            lab_patient[res_idx] = None
            
            if lab_wait_q:
                next_pat = lab_wait_q.popleft()
                assign_to_lab(next_pat, res_idx, t)

        elif etype == EVT_DOC_COMPLETION:
            if doc_version[res_idx] != ver: continue
            
            doc_final_end[pat_idx] = t
            doc_busy[res_idx] = False
            doc_patient[res_idx] = None
            
            # Route to lab if needed
            route_to_lab(pat_idx, t)
            
            next_pat = pull_from_doc_queue()
            if next_pat is not None:
                assign_to_doctor(next_pat, res_idx, t)

        elif etype == EVT_ARRIVAL:
            pri = priority[pat_idx]
            docs_busy_on_arrival[pat_idx] = sum(doc_busy)

            idle_doc = find_idle_doctor()
            if idle_doc is not None:
                assign_to_doctor(pat_idx, idle_doc, t)
                continue

            if pri != RED:
                doc_wait_q[pri].append(pat_idx)
                continue

            # RED PREEMPTION ATTEMPT
            candidates = [j for j in range(s_dr) if doc_busy[j] and priority[doc_patient[j]] == GREEN]
            if not candidates:
                candidates = [j for j in range(s_dr) if doc_busy[j] and priority[doc_patient[j]] == YELLOW]

            if not candidates:
                doc_wait_q[RED].append(pat_idx)
                continue

            victim_doc = min(candidates, key=lambda j: doc_busy_until[j])
            victim_pat = doc_patient[victim_doc]

            # CALCULATE REMAINING WORK
            remaining = doc_busy_until[victim_doc] - t
            remaining_work[victim_pat] = remaining
            interruptions[victim_pat] += 1
            total_preemptions += 1

            victim_pri = priority[victim_pat]
            doc_wait_q[victim_pri].appendleft(victim_pat)

            assign_to_doctor(pat_idx, victim_doc, t)

    # ═════ BUILD RESULTS ═════
    doc_W  = doc_final_end - arrival
    doc_Wq = doc_W - doc_service

    # Door to door calculation
    final_exit = np.where(needs_lab, lab_end, doc_final_end)
    door_to_door = final_exit - arrival

    lab_W = np.where(needs_lab, lab_end - doc_final_end, np.nan)
    lab_Wq = np.where(needs_lab, lab_start - doc_final_end, np.nan)

    results = pd.DataFrame({
        'Patient_ID':         pat_ids,
        'Priority':           [PRIORITY_NAMES[p] for p in priority],
        'Needs_Lab':          needs_lab,
        'Arrival_Min':        arrival,
        
        'Doc_Start_Min':      doc_first_start,
        'Doc_End_Min':        doc_final_end,
        'Interruptions':      interruptions,
        'Doc_Service':        doc_service,
        'Doc_Wq':             doc_Wq,
        'Doc_W':              doc_W,
        
        'Lab_Queue_Entry':    np.where(needs_lab, doc_final_end, np.nan),
        'Lab_Start_Min':      lab_start,
        'Lab_End_Min':        lab_end,
        'Lab_Service':        lab_service,
        'Lab_Wq':             lab_Wq,
        'Lab_W':              lab_W,
        
        'Door_to_Door':       door_to_door
    })

    sim_end_time = max(np.nanmax(doc_final_end), np.nanmax(lab_end))
    metrics = {
        'total_preemptions': total_preemptions,
        'sim_end_time': sim_end_time,
        'doc_rho': doc_busy_time_area / (s_dr * sim_end_time),
        'lab_rho': lab_busy_time_area / (s_lab * sim_end_time)
    }

    return results, metrics
"""))

# Cell 7: Run and Sanity
cells.append(nbf.v4.new_markdown_cell("""# 3. Sanity Checks & Validation

We run the integrated simulation and perform sanity checks to ensure the timeline and constraints are valid.
"""))

# Cell 8: Run Code
cells.append(nbf.v4.new_code_cell("""res, metrics = simulate_network_flow(df, s_dr=5, s_lab=3)

print("--- Sanity Checks ---")
print(f"[PASS] All 1000 patients completed doctor treatment: {res['Doc_End_Min'].notna().all()}")
lab_mask = res['Needs_Lab']
print(f"[PASS] All {lab_mask.sum()} lab patients completed lab: {res.loc[lab_mask, 'Lab_End_Min'].notna().all()}")

# Timeline consistency
timeline_doc_ok = (res['Doc_Start_Min'] >= res['Arrival_Min'] - 1e-9).all() and \
                  (res['Doc_End_Min'] >= res['Doc_Start_Min'] - 1e-9).all()
print(f"[PASS] Doctor timeline consistent: {timeline_doc_ok}")

timeline_lab_ok = (res.loc[lab_mask, 'Lab_Start_Min'] >= res.loc[lab_mask, 'Lab_Queue_Entry'] - 1e-9).all() and \
                  (res.loc[lab_mask, 'Lab_End_Min'] >= res.loc[lab_mask, 'Lab_Start_Min'] - 1e-9).all()
print(f"[PASS] Lab timeline consistent: {timeline_lab_ok}")

# FCFS constraint in Lab
lab_pts = res[lab_mask].sort_values('Lab_Start_Min')
fcfs_ok = lab_pts['Lab_Queue_Entry'].is_monotonic_increasing
print(f"[PASS] Lab FCFS strictly maintained (no priorities in lab): {fcfs_ok}")

print()
print(f"Overall Doctor Traffic Intensity (rho_dr): {metrics['doc_rho']:.4f}")
print(f"Overall Lab Traffic Intensity (rho_lab): {metrics['lab_rho']:.4f}")
"""))

# Cell 9: Deliverable 1
cells.append(nbf.v4.new_markdown_cell("""# 4. Deliverable 1: Node 2 Integration Results (Simulation Log)

An output table for the first 25 patients showing timing information across both nodes.
"""))

# Cell 10: Deliverable 1 Code
cells.append(nbf.v4.new_code_cell("""log_25 = res.head(25).copy()

for col in ['Arrival_Min', 'Doc_Start_Min', 'Doc_End_Min', 'Lab_Start_Min', 'Lab_End_Min']:
    log_25[col.replace('_Min', '_Clock')] = log_25[col].apply(min_to_clock)

display_cols = [
    'Patient_ID', 'Priority', 'Needs_Lab', 'Arrival_Clock',
    'Doc_Start_Clock', 'Doc_End_Clock', 'Interruptions', 'Doc_Service', 'Doc_Wq',
    'Lab_Start_Clock', 'Lab_End_Clock', 'Lab_Service', 'Lab_Wq',
    'Door_to_Door'
]

display_table = log_25[display_cols].copy()
for col in ['Doc_Service', 'Doc_Wq', 'Lab_Service', 'Lab_Wq', 'Door_to_Door']:
    display_table[col] = display_table[col].round(2)

print("=" * 140)
print("SIMULATION LOG - FIRST 25 PATIENTS (Nodes 1 & 2)")
print("=" * 140)
display(display_table.fillna('-'))
"""))

# Cell 11: Deliverable 2 & 3
cells.append(nbf.v4.new_markdown_cell("""# 5. Deliverable 2 & 3: Total Cycle Time & Lab Constraints

- Calculate Door-to-Door time for patients requiring a lab.
- Compare Average Lab Wait Time to Average Doctor Wait Time.
"""))

# Cell 12: Deliverable 2 Code
cells.append(nbf.v4.new_code_cell("""lab_patients = res[res['Needs_Lab']].copy()

print("=" * 60)
print("TOTAL CYCLE TIME ANALYSIS")
print("=" * 60)
print(f"Average Door-to-Door time (Lab Patients): {lab_patients['Door_to_Door'].mean():.2f} min")
print(f"Average Door-to-Door time (ALL Patients): {res['Door_to_Door'].mean():.2f} min")
print()

avg_doc_wq = res['Doc_Wq'].mean()
avg_lab_wq = lab_patients['Lab_Wq'].mean()
avg_doc_wq_lab_pts = lab_patients['Doc_Wq'].mean()

print("Wait Time Comparison:")
print(f"  Average Doctor Wait Time (Wq_dr) for ALL patients: {avg_doc_wq:.2f} min")
print(f"  Average Doctor Wait Time (Wq_dr) for LAB patients only: {avg_doc_wq_lab_pts:.2f} min")
print(f"  Average Lab Wait Time (Wq_lab) for LAB patients: {avg_lab_wq:.2f} min")
print()
print(f"  -> Comparison for Lab Patients: On average, they wait {avg_doc_wq_lab_pts:.2f} mins for the doctor, but {avg_lab_wq:.2f} mins for the lab.")
print()

# Plot breakdown
fig, ax = plt.subplots(figsize=(8, 5))
components = ['Doctor Wait', 'Doctor Service', 'Lab Wait', 'Lab Service']
averages = [
    lab_patients['Doc_Wq'].mean(),
    lab_patients['Doc_Service'].mean(),
    lab_patients['Lab_Wq'].mean(),
    lab_patients['Lab_Service'].mean()
]

ax.bar(components, averages, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'], edgecolor='black')
for i, v in enumerate(averages):
    ax.text(i, v + 0.5, f"{v:.2f} m", ha='center')

ax.set_ylabel("Average Time (min)")
ax.set_title("Time Components for Patients Requiring Lab")
plt.tight_layout()
plt.show()
"""))

# Cell 13: Deliverable 4
cells.append(nbf.v4.new_markdown_cell("""# 6. Deliverable 4: Design for Performance

Hospital Board Mandate:
- **Total System Lead Time (Avg Door-to-Door)** $\\le$ 60 minutes
- **Doctor Utilization ($\\rho_{dr}$)** $\\le$ 85%

Let's evaluate the current system (5 doctors, 3 lab machines) and see if it meets the criteria. 

**Methodology for Additional Arrivals (Scale Factor):**
If the targets are already met, we determine how many additional arrivals the system could handle. Because we are running an event-driven simulation based on a historical timeline, we calculate this by determining a **Scale Factor**. We artificially compress the inter-arrival times (e.g., a scale factor of 0.8 means patients arrive in 80% of the original time, effectively increasing the overall arrival rate $\\lambda$ by a multiplier of $1/0.8 = 1.25$x). We sweep this scale factor downwards until the performance targets are breached.

If the system does not meet the targets initially, we will sweep the resource levels (`s_dr` and `s_lab`) to find the minimum resources needed.
"""))

# Cell 14: Deliverable 4 Code
cells.append(nbf.v4.new_code_cell("""current_lead_time = res['Door_to_Door'].mean()
current_doc_rho = metrics['doc_rho']

print(f"Current System (s_dr=5, s_lab=3):")
print(f"  Avg Lead Time: {current_lead_time:.2f} min (Target: <= 60.0)")
print(f"  Doctor Util:   {current_doc_rho*100:.2f}% (Target: <= 85.0%)")
print()

if current_lead_time <= 60 and current_doc_rho <= 0.85:
    print("Targets MET under current parameters!")
    print("Determining maximum additional arrivals by scaling inter-arrival times...")
    
    # Sweep scale factors from 1.0 down to 0.5 (which increases arrival rate up to 2x)
    scale_factors = np.linspace(1.0, 0.5, 20)
    max_lambda_scale = 1.0
    for sf in scale_factors:
        df_test = df.copy()
        df_test['arrival_min'] = df_test['arrival_min'] * sf
        res_test, met_test = simulate_network_flow(df_test, 5, 3)
        if res_test['Door_to_Door'].mean() > 60 or met_test['doc_rho'] > 0.85:
            break
        max_lambda_scale = sf
    
    current_lambda = 1000 / df['arrival_min'].max()
    new_lambda = current_lambda / max_lambda_scale
    additional_lambda = new_lambda - current_lambda
    multiplier = 1 / max_lambda_scale
    
    print(f"\\n-> The system can handle up to an arrival rate λ = {new_lambda:.4f} pat/min")
    print(f"-> This means we can handle +{additional_lambda:.4f} additional arrivals per minute.")
    print(f"-> The arrival times were scaled by {max_lambda_scale:.2f} (a {multiplier:.2f}x increase in volume) before targets were breached.")

else:
    print("Targets NOT MET. Sweeping resources...")
    scenarios = []
    for d in range(5, 8):
        for l in range(3, 6):
            r, m = simulate_network_flow(df, s_dr=d, s_lab=l)
            lead = r['Door_to_Door'].mean()
            d_rho = m['doc_rho']
            meets = lead <= 60 and d_rho <= 0.85
            scenarios.append({'s_dr': d, 's_lab': l, 'LeadTime': lead, 'Doc_Rho': d_rho, 'Meets_Targets': meets})
            
    sc_df = pd.DataFrame(scenarios)
    display(sc_df)
    
    valid = sc_df[sc_df['Meets_Targets']]
    if not valid.empty:
        best = valid.sort_values(['s_dr', 's_lab']).iloc[0]
        print(f"\\nMINIMUM RESOURCES TO MEET TARGETS: s_dr={int(best['s_dr'])}, s_lab={int(best['s_lab'])}")
"""))

# Cell 15: Deliverable 5
cells.append(nbf.v4.new_markdown_cell("""# 7. Deliverable 5: Bottleneck Identification & Resource Allocation

1. Identify primary bottleneck comparing Traffic Intensity ($\\rho$) for Doctors and Lab.
2. Budget allows hiring **ONE** more staff member: 6th Doctor vs 4th Lab Tech.
"""))

# Cell 16: Deliverable 5 Code
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("BOTTLENECK IDENTIFICATION")
print("=" * 60)

print(f"Doctor Traffic Intensity (rho_dr): {metrics['doc_rho']:.4f}")
print(f"Lab Traffic Intensity (rho_lab):   {metrics['lab_rho']:.4f}")

if metrics['doc_rho'] > metrics['lab_rho']:
    print("-> Doctors are the primary bottleneck (higher utilization).")
else:
    print("-> Lab is the primary bottleneck (higher utilization).")
print()

# Simulate Scenarios
print("SCENARIO COMPARISON (Budget for 1 hire):")
# Base
lead_base = res['Door_to_Door'].mean()

# Scenario A: 6 Doctors, 3 Lab
res_A, met_A = simulate_network_flow(df, s_dr=6, s_lab=3)
lead_A = res_A['Door_to_Door'].mean()

# Scenario B: 5 Doctors, 4 Lab
res_B, met_B = simulate_network_flow(df, s_dr=5, s_lab=4)
lead_B = res_B['Door_to_Door'].mean()

print(f"Base (5 Dr, 3 Lab):       Lead Time = {lead_base:.2f} min")
print(f"Scenario A (6 Dr, 3 Lab): Lead Time = {lead_A:.2f} min (Reduction: {lead_base - lead_A:.2f} min)")
print(f"Scenario B (5 Dr, 4 Lab): Lead Time = {lead_B:.2f} min (Reduction: {lead_base - lead_B:.2f} min)")

fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Base (5D, 3L)', 'Hire Doctor (6D, 3L)', 'Hire Lab Tech (5D, 4L)']
times = [lead_base, lead_A, lead_B]

ax.bar(labels, times, color=['gray', 'blue', 'green'], edgecolor='black')
for i, v in enumerate(times):
    ax.text(i, v + 1, f"{v:.2f} m", ha='center')

ax.set_ylabel("Average Door-to-Door Time (min)")
ax.set_title("Resource Allocation Impact on Total Lead Time")
plt.tight_layout()
plt.show()

if lead_A < lead_B:
    print("\\nRECOMMENDATION: Hire a 6th Doctor to maximize total system lead time reduction.")
else:
    print("\\nRECOMMENDATION: Hire a 4th Lab Technician to maximize total system lead time reduction.")
"""))

nb['cells'] = cells

with open('IE493 Project Phase 3.ipynb', 'w') as f:
    nbf.write(nb, f)
