# -*- coding: utf8

import gzip
import json
import networkx as nx
import numpy as np



def car(A, B):
    C=list()
    for i in A:
        for j in B:
            C.append((i,j))
    return set(C)

def normalize(array):
  if type(array) == type(dict()):
    soma = sum(array.values())
    for i in array.keys():
      array[i] = array[i]/soma
    return array
  return np.array(array)/np.sum(array)

def non_zero_valid_indices(lista):
  return np.array([i for i in range(len(lista)) if not np.isnan(lista[i]) and lista[i]!= 0])

def valid_indices(lista):
  return np.array([i for i in range(len(lista)) if not np.isnan(lista[i])])



def surprise(pMs, data, hipoteses, years_range = None):
  if years_range == None:
    years_range = range(data.shape[0])

  pMs = np.array(pMs)
  surpriseData = np.zeros(data.shape)

  lista_pM = [pMs]

  pDMs = np.zeros(pMs.shape)
  pMDs = np.zeros(pMs.shape)

  diferencas = np.zeros(pMs.shape)
  soma_das_diferencas = np.zeros(pMs.shape)

  for ano in years_range:
    #i = ano
    valid_artist_indices = non_zero_valid_indices(data[ano-1,:])


    soma_das_diferencas = np.zeros(pMs.shape)
    vetor_diferencas = np.zeros((len(pMs),len(data[ano, :][valid_artist_indices])))

    for i in range(len(diferencas)):
      vetor_diferencas[i] = data[ano, :][valid_artist_indices] - normalize(hipoteses[i](data, ano)[:len(valid_artist_indices)])

    for artista in range(len(valid_artist_indices)):

      for i in range(len(diferencas)):
        diferencas[i] = vetor_diferencas[i,artista]
        pDMs[i] = 1 - np.abs(diferencas[i])


      #Estimate P(M|D)
      #uniform
      #P(M|D) = P(M) * P(D|M)

      pMDs = pMs*pDMs


      kl = 0;
      voteSum = 0;
      for j in range(len(pMDs)):
        kl = kl + pMDs[j] * (np.log( pMDs[j] / pMs[j]) / np.log(2)) #É essa a fórmula mesmo?
        voteSum = voteSum + diferencas[j] * pMs[j]
        soma_das_diferencas[j] = soma_das_diferencas[j] + np.abs(diferencas[j])


      if voteSum >= 0 :
        surpriseData[ano, artista] = np.abs(kl)
      else:
        surpriseData[ano, artista] = -1*np.abs(kl)

    #Now lets globally update our model belief.

    for j in range(len(pMs)):
      pDMs[j] = 1 - (0.5 * soma_das_diferencas[j])
      pMDs[j] = pMs[j] * pDMs[j]
      pMs[j] = pMDs[j]

    #Normalize
    # for j in range(len(pMs)):
    #   pMs[j] = pMs[j] / sum(pMs)
    pMs = normalize(pMs)
    print(pMs)

    lista_pM.append(pMs)
  return surpriseData, lista_pM


#Parametro: um sample e
#Retorno: artista sampleado e artistas que samplearam
def extract_who_sampled(sample):
    who_sampled = set()
    who_was_sampled = set()
    for key in sample.keys():
        if key[:3] == 'was':
            for artist in sample[key]['other_track_by']:
                who_sampled.add(artist['name'])
            who_was_sampled = sample[key]['by'][0]['name']
        else:
            for artist in sample[key]['by']:
                who_sampled.add(artist['name'])
            who_was_sampled = sample[key]['other_track_by'][0]['name']
    return who_was_sampled, who_sampled


#here
def extract_track_who_sampled(sample):
    for key in sample.keys():
        if key[:3] == 'was':
            who_sampled = sample[key]['other_track']['name']
            who_was_sampled = sample[key]['track']['name']
        else:
            who_sampled = sample[key]['track']['name']
            who_was_sampled = sample[key]['other_track']['name']
    return who_was_sampled, who_sampled




def extract_id(txt):
    pos = txt.rfind('mn')
    return txt[pos:pos+12]


def build_reverse_index(json_data):
    decades = {}
    genres = {}
    styles = {}
    for key, value in json_data.items():
        for genre_id, genre_name in value['genres'].items():
            if genre_name not in genres:
                genres[genre_name] = set()
            genres[genre_name].add(key)
            break

        for style_id, style_name in value['styles'].items():
            if style_name not in styles:
                styles[style_name] = set()
            styles[style_name].add(key)
            break

        if value['decades']:
            decade = value['decades'][0]
            if decade not in decades:
                decades[decade] = set()
            decades[decade].add(key)
    return decades, genres, styles


def build_graph(json_data, edges_to_consider=None,
                edge_consideration_criterion='genres',
                nodes_to_consider=None,
                restrictive=False,
                decades_to_consider = None):

    G = nx.DiGraph()
    if nodes_to_consider is None:
        nodes_to_consider = set(json_data.keys())

    if decades_to_consider!= None:
      nodes_to_consider_filter=[]
      for node in nodes_to_consider:
          if set(json_data[node]['decades']).intersection(set(decades_to_consider)):
              nodes_to_consider_filter.append(node)
      nodes_to_consider = set(nodes_to_consider_filter)


    for artist in nodes_to_consider:
        data = json_data[artist]
        by_set = set(map(extract_id, data['influencer']))
        for by in by_set:
            if restrictive and by not in nodes_to_consider:
                continue
            if decades_to_consider!=None and not set(json_data[by]['decades']).intersection(set(decades_to_consider)):
                continue

            if edges_to_consider is None:
                if G.has_edge(artist, by):
                    G[artist][by]['weight'] += 1
                else:
                    G.add_weighted_edges_from([(artist, by, 1)])
                continue
            if list(json_data[artist][edge_consideration_criterion].values())!=[] and list(json_data[by][edge_consideration_criterion].values())!=[]:
                dupla = list(json_data[artist][edge_consideration_criterion].values())[0], list(json_data[by][edge_consideration_criterion].values())[0]
                if dupla in edges_to_consider:
                    G.add_edge(artist, by)

    nx.set_node_attributes(G, json_data)
    return G


def build_graph_who_sampled(json_data, edges_to_consider=None):
    G = nx.DiGraph()
    if edges_to_consider==None:
      edges_to_consider = json_data
    for sample in edges_to_consider:
        data = json_data[sample]
        who_was_sampled, who_sampled = extract_who_sampled(data)
        for artist_that_sampled in who_sampled:
            if G.has_edge(artist_that_sampled, who_was_sampled):
                  G[artist_that_sampled][who_was_sampled]['weight'] += 1
            else:
                   G.add_weighted_edges_from([(artist_that_sampled, who_was_sampled, 1)])

            G.add_edge
    nx.set_node_attributes(G, json_data)
    return G

#here
def build_track_graph_who_sampled(json_data):
    G = nx.DiGraph()
    for sample in json_data:
        data = json_data[sample]
        who_was_sampled, who_sampled = extract_track_who_sampled(data)
        G.add_edge(who_sampled, who_was_sampled)
    #nx.set_node_attributes(G, json_data)
    return G

#This function build a graph where each node is a genre, and each edge and its weight represent
#the number of artists from the outgoing node genre, have influencers in the incoming node genre.
def build_genre_graph(json_data):
    G= nx.DiGraph()
    if nodes_to_consider is None:
        nodes_to_consider = set(json_data.keys())

    for artist in nodes_to_consider:
        artist_node = json_data[artist]
        influencer_set = set(map(extract_id, artist_node['influencer'])) # lista com todos os ids dos influencer do artist_note
        try:
            u = next(iter(json_data[artist]['genres'].values()))
            for influencer in influencer_set:
                try:
                    v=next(iter(json_data[influencer]['genres'].values()))
                    if  u!=v:
                        if not G.has_edge(u,v):
                            G.add_edge(next(iter(json_data[artist]['genres'].values())), next(iter(json_data[influencer]['genres'].values())), weight= 1)
                        else:
                            new_weight = int(G.get_edge_data(u,v)['weight']) + 1
                            G.remove_edge(u,v)
                            G.add_edge(u,v, weight=new_weight )

                except:
                    continue
        except:
            continue
    return G


def load_am_json_data():
    fpath = '../data/allmusic-data.json.gz'
    with gzip.open(fpath) as gzip_file:
        json_data = json.load(gzip_file)
        return json_data
