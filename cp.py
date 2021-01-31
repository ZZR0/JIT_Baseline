import os

# cmd = '\cp data/{}/cross/hk_test.csv data/{}/cross/hk_test_back.csv'
# cmd2 = '\cp data/{}/hk_train.csv data/{}/cross/hk_test.csv'


# projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
# for project in projects:
#     os.system(cmd.format(project, project))
#     os.system(cmd2.format(project, project))


cmd = '\cp data/{}/cross/hk_test_back.csv data/{}/cross/hk_test.csv'

projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
for project in projects:
    os.system(cmd.format(project, project))