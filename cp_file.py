import os

cmd_lr = '\cp data/{}/{}_lr.result result/{}_lr.result'
cmd_dbn = '\cp data/{}/{}_dbn.result result/{}_dbn.result'
cmd_sel = '\cp data/{}/{}_sel.result result/{}_sel.result'

for project in ['gerrit', 'go', 'jdt', 'qt', 'openstack', 'platform']:
    # os.system(cmd_lr.format(project, project, project))
    # os.system(cmd_dbn.format(project, project, project))
    os.system(cmd_sel.format(project, project, project))



