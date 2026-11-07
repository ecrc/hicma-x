import pandas as pd

# Input data
t = pd.HDFStore('./timing_kernel3-lfq-O6ulKS.h5')

t.events = t.events.fillna(-1)
t.events[['tid', 'did', 'tpid', 'node_id', 'id', 'src', 'dst', 'm', 'k']] = t.events[['tid', 'did', 'tpid', 'node_id', 'id', 'src', 'dst', 'm', 'k']].astype(int)

tasks = t.events[t.events.type >= t.event_types['PUT_CB']+1]
tasks.set_index(['type', 'm', 'k'], inplace=True) # for normal task, use id field for ID

seq_time = 0.
seq_time = sum(tasks.loc[t.event_types['potrf_dpotrf'], :, :].end - tasks.loc[t.event_types['potrf_dpotrf'], :, :].begin)/1.e9

for i in range(1, tasks.loc[t.event_types['potrf_dpotrf']].shape[0]):
    if i % 50 == 0:
        print i
    seq_time +=  float((tasks.loc[t.event_types['potrf_dtrsm'], i, i-1].end - \
                 tasks.loc[t.event_types['potrf_dtrsm'], i, i-1].begin)/1.e9)
    seq_time +=  float((tasks.loc[t.event_types['potrf_dsyrk'], i, i-1].end - \
                 tasks.loc[t.event_types['potrf_dsyrk'], i, i-1].begin)/1.e9)

print seq_time
