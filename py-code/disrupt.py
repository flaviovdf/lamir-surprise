# -*- coding: utf8

import networkx as nx
import numpy as np
import pandas as pd
import plac

def get_confidence_disruption(disrupt, prior=10):
    confidences = []
    for ni, nj, nk, disruption in disrupt[['ni', 'nj', 'nk', 'disruption']].values:
        D = np.random.dirichlet([prior + ni, prior + nj, prior + nk], size=10000)
        pos_i = D[:, 0]
        pos_j = D[:, 1]
        if disruption <= 0:
            confidence = ((D[:, 0] - D[:, 1]) < 0).mean()
        else:
            confidence = ((D[:, 0] - D[:, 1]) > 0).mean()
        confidences.append(confidence)
    return np.array(confidences)


def compute_disruption(G, min_in=1, min_out=0):

    id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    in_count = dict(G.in_degree(G.nodes))
    out_count = dict(G.out_degree(G.nodes))

    F = nx.to_scipy_sparse_matrix(G, format='csr')
    T = nx.to_scipy_sparse_matrix(G, format='csc')
    D = np.zeros(shape=(F.shape[0], 6))

    for node_id in range(F.shape[0]):
        if in_count[id_to_node[node_id]] >= min_in and \
                out_count[id_to_node[node_id]] >= min_out:
            ni = 0
            nj = 0
            nk = 0

            outgoing = F[node_id].nonzero()[1]
            incoming = T[:, node_id].nonzero()[0]
            outgoing_set = set(outgoing)

            for other_id in incoming:
                second_level = F[other_id].nonzero()[1]
                if len(outgoing_set.intersection(second_level)) == 0:
                    ni += 1
                else:
                    nj += 1

            # who mentions my influences
            who_mentions_my_influences = np.unique(T[:, outgoing].nonzero()[0])
            for other_id in who_mentions_my_influences:
                # do they mention me?! if no, add nk
                if F[other_id, node_id] == 0 and other_id != node_id:
                    nk += 1

            D[node_id, 0] = ni
            D[node_id, 1] = nj
            D[node_id, 2] = nk
            D[node_id, 3] = (ni - nj) / (ni + nj + nk)
            D[node_id, 4] = in_count[id_to_node[node_id]]
            D[node_id, 5] = out_count[id_to_node[node_id]]
        else:
            D[node_id, 0] = np.nan
            D[node_id, 1] = np.nan
            D[node_id, 2] = np.nan
            D[node_id, 3] = np.nan
            D[node_id, 4] = in_count[id_to_node[node_id]]
            D[node_id, 5] = out_count[id_to_node[node_id]]

    return pd.DataFrame(D, index=G.nodes,
                        columns=['ni', 'nj', 'nk', 'disruption', 'in', 'out'])


# def compute_disruption(G, separation_criterion = 'genres', min_in=1, min_out=0, time_consideration = False):

#     # Cria um dicionario onde pra cada número eu tenho um nó do grafo
#     id_to_node = dict((i, n) for i, n in enumerate(G.nodes))
    
#     # Número de arestas apontando para o node
#     in_count = dict(G.in_degree(G.nodes))
    
#     # Número de arestas saindo do node
#     out_count = dict(G.out_degree(G.nodes))

#     # Guarda em F a matriz esparsa do grafo no formato compressed sparse row
#     F = nx.to_scipy_sparse_matrix(G, format='csr') 
    
#     # Guarda em T a matriz esparsa do grafo no formato compressed sparse column
#     T = nx.to_scipy_sparse_matrix(G, format='csc')
    
#     # Cria uma matriz onde o número de linhas é o número de nodes do grafo e o número de colunas é igual a 6
#     # Essa matriz será transformada em um DataFrame pandas
#     # Cada linha dessa matriz é um node e cada indice dessa linha é uma informação sobre esse node 
#     D = np.zeros(shape=(F.shape[0], 12))

#     # Itera as linhas da matriz ou seja, uma iteração para cada node
#     for node_id in range(F.shape[0]):
#         #Só pra saciar minha ansiedade
#         if node_id%2000==0:
#             print(node_id/2000)
#         #Se o número de in_edges do node correspondente a essa linha for maior q o número mínimo aceito de in_edges e
#         #Se o número de out_edges for maior q o número mínimo aceito de out_edges.
#         if in_count[id_to_node[node_id]] >= min_in and \
#                 out_count[id_to_node[node_id]] >= min_out:
#             #ni: número de followers exclusivos do node em realação ao seus influencers
#             #nj: número de followers do node q também são followers dos influencers do node
#             #nk: número de followers exclusivos dos influencers do node, em relação a ele
#             ni_E = 0
#             ni_D = 0
#             nj_E = 0
#             nj_D = 0
#             nk_E = 0
#             nk_D = 0
            
#             # Indices da linha node_id que contêm 1
#             # Ou seja, os node_ids dos nodes pelos quais o node_id atual aponta
#             outgoing = F[node_id].nonzero()[1]
            
#             # Indices da coluna node_id que contêm 1         
#             # Ou seja, os node_ids dos nodes que apontam para o node_id atual
#             incoming = T[:, node_id].nonzero()[0]
            
#             # Set do outgoing? Já não possuia apenas elementos distintos?  ######### Pergunta #########
#             outgoing_set = set(outgoing)
        
            
            
#             for other_id in incoming: # Id dos followers, nodes q apontontam para o node desse laço
#                 second_level = F[other_id].nonzero()[1] # Nodes pelos quais o node other_id aponta
                
#                 node_criterion = 'Sem '+ separation_criterion
#                 other_criterion = 'Sem '+ separation_criterion
#                 if G.nodes[id_to_node[node_id]][separation_criterion] !={}:
#                     node_criterion  = next(iter(G.nodes[id_to_node[node_id]][separation_criterion].values()))
#                 if G.nodes[id_to_node[other_id]][separation_criterion] !={}:
#                     other_criterion = next(iter(G.nodes[id_to_node[other_id]][separation_criterion].values()))
                
#                 # Se o other id não aponta pra mais ninguém
#                 if len(outgoing_set.intersection(second_level)) == 0: 
#                     if node_criterion == other_criterion: 
#                         ni_E += 1
#                     else:
#                         ni_D += 1
                    
#                 # Se ele apontar para outros
#                 else:
#                     if node_criterion == other_criterion: 
#                         nj_E += 1
#                     else:
#                         nj_D += 1
                    
                
#             # Quem menciona minhas influências
#             who_mentions_my_influences = np.unique(T[:, outgoing].nonzero()[0]) # Followers dos caras que eu sigo
            
#             for other_id in who_mentions_my_influences: # Para cada Follower dos caras q eu sigo
#                 if F[other_id, node_id] == 0 and other_id != node_id: # Se eles seguem o cara e eles não são eu
#                     node_criterion = 'Sem '+ separation_criterion
#                     other_criterion = 'Sem '+ separation_criterion
#                     if G.nodes[id_to_node[node_id]][separation_criterion] !={}:
#                         node_criterion  = next(iter(G.nodes[id_to_node[node_id]][separation_criterion].values()))
#                     if G.nodes[id_to_node[other_id]][separation_criterion] !={}:
#                         other_criterion = next(iter(G.nodes[id_to_node[other_id]][separation_criterion].values()))
#                     if time_consideration == True:
                        
#                         if G.nodes[id_to_node[node_id]]['decades'] != []:
#                             time_node = G.nodes[id_to_node[node_id]]['decades'][0]
#                         else:
#                             if 'birthdate.id' in G.nodes[id_to_node[node_id]].keys():
#                                 time_node = int(G.nodes[id_to_node[node_id]]['birthdate.id'][0:4])
#                             else:
#                                 continue
                            
#                         if G.nodes[id_to_node[other_id]]['decades'] != []:
#                             time_other = G.nodes[id_to_node[other_id]]['decades'][-1]
                                
#                         else:
#                             if 'deathdate.id' in G.nodes[id_to_node[node_id]].keys():
#                                 time_other = int(G.nodes[id_to_node[node_id]]['deathdate.id'][0:4])
#                             else:
#                                 continue
                            
#                         if time_other>time_node:
#                             if node_criterion == other_criterion:
#                                 nk_E += 1
#                             else:
#                                 nk_D += 1
#                     else:
#                         if node_criterion == other_criterion:
#                             nk_E += 1
#                         else:
#                             nk_D += 1
                        
#             #Preenchendo a matriz q vai ser transformada em um DataFrame pandas
            
#             D[node_id, 0] = ni_E
#             D[node_id, 1] = ni_D
#             D[node_id, 2] = ni_E + ni_D
#             D[node_id, 3] = nj_E
#             D[node_id, 4] = nj_D
#             D[node_id, 5] = nj_E + nj_D
#             D[node_id, 6] = nk_E
#             D[node_id, 7] = nk_D
#             D[node_id, 8] = nk_E + nk_D
#             D[node_id, 9] = ((ni_E + ni_D) - (nj_E + nj_D)) / (ni_E + ni_D + nj_E + nj_D + nk_E + nk_D)
#             D[node_id, 10] = in_count[id_to_node[node_id]] #Número de arestas apontando para o node desse laço
#             D[node_id, 11] = out_count[id_to_node[node_id]]#Número de nodes q são apontados pelo node desse laço
            
            
#         else: # Caso não tenha in_edges e out_edges o suficiente
#             D[node_id, 0] = np.nan
#             D[node_id, 1] = np.nan
#             D[node_id, 2] = np.nan
#             D[node_id, 3] = np.nan
#             D[node_id, 4] = np.nan
#             D[node_id, 5] = np.nan
#             D[node_id, 6] = np.nan
#             D[node_id, 7] = np.nan
#             D[node_id, 8] = np.nan
#             D[node_id, 9] = np.nan
#             D[node_id, 10] = in_count[id_to_node[node_id]]
#             D[node_id, 11] = out_count[id_to_node[node_id]]

#     return pd.DataFrame(D, index=G.nodes,
#                         columns=['ni_E', 'ni_D', 'ni', 'nj_E', 'nj_D', 'nj', 'nk_E', 'nk_D', 'nk', 'disruption', 'in', 'out'])


def main(input_graph: ('Input file, list of edges', 'positional', None, str),
         output_file: ('Output file, a csv', 'positional', None, str),
         directed: ('Indicates if the graph is connected', 'flag', 'd')):

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    with open(input_graph) as graph_file:
        for line in graph_file:
            src, dst = line.strip().split()
            G.add_edge(src, dst)

    df = compute_disruption(G)
    df.to_csv(output_file)


if __name__ == '__main__':
    plac.call(main)
