import os
import re

drop_cmd = 'python baseline.py -data cross/hk -project {} -algorithm lr -drop '
only_cmd = 'python baseline.py -data cross/hk -project {} -algorithm lr -only '
sele_cmd = 'python baseline.py -data k -project {} -algorithm sel'
algo_cmd = 'python baseline.py -data cross/hk -project {} -algorithm '
size_cmd = 'python baseline.py -data {}/hk -project {} -algorithm {}'


def run_project(project, features, cmd=drop_cmd):
    results = []
    for key in features:
        result = os.popen(cmd.format(project) + key).readlines()
        results.append((key, result[-1]))
        print(result)
    
    return results

def run_project_size(projects, cmd=drop_cmd):
    results = []
    for project in projects:
        result = os.popen(cmd.format(project)).readlines()
        results.append((project, result[-1]))
        print(result)
    
    return results

def run_size(projects, cmd):
    sizes = ['50k', '40k', '30k', '20k', '10k']
    result_file = 'size_result.csv'

    with open(result_file, 'w', encoding='utf-8') as f:

        for al in ['dbn']:
            auc_result = [[project] for project in projects]

            for size in sizes:
                n_cmd = cmd.format(size, '{}', al)
                results = run_project_size(projects, cmd=n_cmd)

                pattern = 'AUC: (\d+.\d+)'
                for idx, line in enumerate(results):
                    key, result = line
                    auc = re.findall(pattern, result)
                    auc = float(auc[0])
                    auc_result[idx].append(auc)
            
            line = al
            for size in sizes:
                line += ', ' + size
            line += '\n'
            f.writelines(line)

            for result in auc_result:
                line = ''
                for auc in result:
                    line += str(auc) + ', '
                line += '\n'
                f.writelines(line)

def run_all(projects, features, cmd):
    auc_result = [[feature] for feature in features]
    rate_result = [[feature] for feature in features]

    if cmd == drop_cmd:
        result_file = 'drop_result.csv'
    elif cmd == only_cmd:
        result_file = 'only_result.csv'
    elif cmd == algo_cmd:
        result_file = 'algo_result.csv'
    elif cmd == sele_cmd:
        result_file = 'sele_result.csv'
    else:
        result_file = 'algo_result.csv'

    for p_idx, project in enumerate(projects):
        # ori_auc = [0.68728,0.724766,0.697913,0.756423,0.776647,0.684281][p_idx]
        results = run_project(project, features, cmd=cmd)

        pattern = 'AUC: (\d+.\d+)'
        for idx, line in enumerate(results):
            key, result = line
            auc = re.findall(pattern, result)
            auc = float(auc[0])
            auc_result[idx].append(auc)

            ori_auc = auc_result[0][-1]
            rate_result[idx].append(str(100*auc/ori_auc))

    with open(result_file, 'w', encoding='utf-8') as f:
        line = ''
        for project in projects:
            line += ', ' + project
        line += '\n'
        f.writelines(line)

        for result in auc_result:
            line = ''
            for auc in result:
                line += str(auc) + ', '
            line += '\n'
            f.writelines(line)

        if cmd != algo_cmd and cmd != sele_cmd:
            for result in rate_result:
                line = ''
                for auc in result:
                    line += str(auc) + ', '
                line += '\n'
                f.writelines(line)


if __name__ == "__main__":
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    # features = ['""', 'ns','nd','nf','ent','la','ld','ld','fix','ndev','age','nuc','aexp','rexp','asexp']
    # features = ['""', 'ns','nd','nf','entrophy','la','ld','lt','fix','ndev','age','nuc','exp','rexp','sexp',
    #             'maxbn','menbn','sumbn','week','week2','month','month3','month6','year']
    features = ['""', 'ns','nd','nf','entrophy','la','ld','lt','fix','ndev','age','nuc','exp','rexp','sexp']
    # features = ['lr', 'dbn', 'la']
    # features = ['lr', 'la']
    # features = ['""']

    run_all(projects, features, only_cmd)
    # run_size(projects, size_cmd)
