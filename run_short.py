import os
import re

short_cmd = 'python baseline.py -data {}/hk -project {} -algorithm {}'

def run_project_short(projects, cmd=short_cmd):
    results = []
    for project in projects:
        # print(cmd.format(project))
        result = os.popen(cmd.format(project)).readlines()
        results.append((project, result[-1]))
        print(result)
    
    return results

def run_short(projects, cmd):
    periods = ['short_1', 'short_2', 'short_3', 'short_4', 'short_5']
    result_file = 'short_result.csv'

    with open(result_file, 'w', encoding='utf-8') as f:

        for al in ['lr', 'la', 'dbn']:
            auc_result = [[project] for project in projects]

            for size in periods:
                n_cmd = cmd.format(size, '{}', al)
                results = run_project_short(projects, cmd=n_cmd)

                pattern = 'AUC: (\d+.\d+)'
                for idx, line in enumerate(results):
                    key, result = line
                    auc = re.findall(pattern, result)
                    auc = float(auc[0])
                    auc_result[idx].append(auc)
            
            line = al
            for size in periods:
                line += ', ' + size
            line += '\n'
            f.writelines(line)

            for result in auc_result:
                line = ''
                for auc in result:
                    line += str(auc) + ', '
                line += '\n'
                f.writelines(line)


if __name__ == "__main__":
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    run_short(projects, short_cmd)
