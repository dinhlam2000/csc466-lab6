import sys
import numpy as np
import time
from scipy.sparse import csr_matrix

def compute_matrix(graph):
    nodes = list(graph.keys())
    try:
        nodes = list(map(int, nodes))
    except:
        nodes = list(graph.keys())
    nodes = sorted(nodes)
    nodes = list(map(str,nodes))

    indices_graph = {}
    rows = []
    cols = []
    for index, node in enumerate(nodes):
        indices_graph[node] = index

    for node in nodes:
        neighbors = graph[node]
        for neighbor in neighbors:
            rows.append(indices_graph[node])
            cols.append(indices_graph[neighbor])

    data = [1]*len(rows)
    matrix_sparse = csr_matrix((data, (rows,cols)), shape=(len(nodes), len(nodes)))
    sums = np.sum(matrix_sparse, axis=1)
    matrix_sparse_prob = matrix_sparse.multiply(1/sums)
    # import pdb; pdb.set_trace()
    return nodes, matrix_sparse_prob.toarray()




def get_vector(g_matrix_transpose: np.ndarray, current_p: [], d : float, ep : float, iteration : int):

    next_p = d*(np.dot(g_matrix_transpose,current_p)) + (1-d)/(len(current_p))
    sum = 0
    for index in range(len(next_p)):
        sum = sum + abs((next_p[index] - current_p[index]))
    if sum < ep:
        # import pdb; pdb.set_trace()
        return next_p, iteration
    else:
        return get_vector(g_matrix_transpose, next_p, d, ep, iteration + 1)







if __name__ == "__main__":

    filename = sys.argv[1]
    d = float(sys.argv[2])
    ep = float(sys.argv[3])


    graph = {}
    read_time_start = time.time()
    with open(filename, 'r') as f:
        all_lines = f.readlines()
        #small data set is csv, while other are .txt
        if '.csv' in filename:
            if "NCAA_football.csv" == filename:
                for line in all_lines:
                    line = line.strip('\n').split(',')
                    if line[2] not in graph:
                        graph[line[2]] = []
                    else:
                        graph[line[2]].append(line[0])
                        graph[line[0]] = []
            else:
                for line in all_lines:
                    line = line.strip('\n').split(',')
                    if line[0] not in graph:
                        graph[line[0]] = []
                    else:
                        graph[line[0]].append(line[2])
                        graph[line[2]] = []


        #construct graph
        elif '.txt' in filename:
            for line in all_lines:
                if line[0] == '#':
                    continue
                line = line.strip('\n').split('\t')
                if line[0] not in graph:
                    graph[line[0]] = []
                else:
                    graph[line[0]].append(line[1])
                    graph[line[1]] = []
    read_time_total = time.time() - read_time_start


    compute_matrix_time_start = time.time()
    keys, matrices = compute_matrix(graph)
    compute_matrix_time_total = time.time() - compute_matrix_time_start

    g_matrix_transpose = matrices.transpose()
    g_matrix_star = []
    for i in range(len(matrices)):
        g_matrix_star.append([1 / len(matrices)] * len(matrices))
    g_matrix_star_transpose = np.array(g_matrix_star).transpose()

    page_rank_time_start = time.time()
    vector, iteration = get_vector(g_matrix_transpose, g_matrix_star_transpose[0], d, ep, 1 )
    page_rank_time_end = time.time() - page_rank_time_start
    indexes_vector = [i[0] for i in sorted(enumerate(vector), key=lambda k: k[1], reverse=True)]

    print('Read Time: ', read_time_total, ' second')
    print('Computing Matrix Time: ', compute_matrix_time_total, ' second')
    print('Computing PageRank Time: ', page_rank_time_end, ' second')
    print('Total Iteration: ', iteration)
    file_out = filename.replace('.csv', '')
    file_out = file_out.replace('.txt', '')

    with open(file_out + 'out.txt', 'w') as f:
        for index in indexes_vector:
            f.write(str(keys[index]) +  ' with page rank: ' + str(vector[index]) + '\n')

