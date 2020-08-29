import pandas as pd
from collections import Counter, defaultdict
from config import LOG_PATH, CSV_PATH
import numpy as np
from bg_logs_reader import parse_logs


def df_to_dict(df):
    dics = []
    for index, match_data in enumerate(df):
        elements = match_data[0].split(',')
        if elements[1] == "":
            continue
        i = 1 if elements[1] in['"Yogg-Saron', '"Aranna'] else 0
        hero = elements[1] if elements[1] != '"Aranna' else "Aranna, Unleashed"
        dics.append({'date': elements[0],
                     'hero': hero,
                     'pos': elements[2+i],
                     'mmr': elements[3+i],
                     'turns': elements[-5:]})
    return dics


def get_all_mmr(data):
    mmr = [int(match_data['mmr'].replace('"', ''))
           for match_data in data if int(match_data['mmr'].replace('"', '')) > 3000]
    return mmr


def get_all_position(data):
    results = defaultdict(list)
    for match in data:
        try:
            results[match['hero']].append(int(match['pos']))
        except:
            pass
    return results


def extract_choices_and_pick(log_path):
    all_new = parse_logs(log_path)  # list, one element per log file
    new = {'choice': [a['choices'] for d in all_new for a in d if a['choices'] != [] and 'The Coin' not in a['choices']],
           'pick':  [a['hero'] for d in all_new for a in d if a['choices'] != [] and 'The Coin' not in a['choices']]}

    return new


def get_mmr_gain(data):
    results = defaultdict(list)
    for index_match, result_match in enumerate(data):
        if index_match == 0:
            results[result_match['hero']].append(0)
        else:
            results[result_match['hero']].append(
                int(result_match['mmr'].replace('"', '')) - int(data[index_match-1]['mmr'].replace('"', '')))
    return {k.replace('"', ''): (np.mean(v), sum(v), -sum([i for i in v if i < 0]), sum([i for i in v if i > 0])) for k, v in results.items()}


def get_pick_stats(new):
    n = {}
    for choice in new['choice']:
        for hero in choice:
            if hero not in n:
                n[hero] = [0, 0]
            n[hero][0] += 1
    for hero in new['pick']:
        n[hero][1] += 1
    return n


def get_top_n_rate(pos):
    results = {k: {i: 0 for i in range(1, 9)} for k in pos.keys()}
    all_pos = [i for poses in pos.values() for i in poses]
    nb_per_pos = Counter(all_pos)
    results['global'] = {k: round(v/len(all_pos), 2)
                         for k, v in nb_per_pos.items()}
    for i in range(1, 9):
        if i not in results['global']:
            results['global'][i] = 0
    for hero, places in pos.items():
        for place in places:
            if place == 0:
                continue
            results[hero][place] += 1
        for k, v in results[hero].items():
            results[hero][k] = round(results[hero][k]/len(pos[hero]), 2)
    return results


def get_all_stats():
    choices_and_pick = extract_choices_and_pick(LOG_PATH)
    picks_stats = get_pick_stats(choices_and_pick)
    df = pd.read_csv(CSV_PATH, sep=';').values
    data = df_to_dict(df)
    all_matches_per_champ = get_all_matches_per_champ(data)
    mmr_evo = get_all_mmr(data)
    positions = get_all_position(data)
    mean_position = round_(
        np.mean([v for x in positions.values() for v in x]), 2)
    mean_pos = {k.replace('"', ''): np.mean(
        v) for k, v in positions.items() if len(v) != 0}  # mean positions
    nb_played = {k.replace('"', ''): len(v)
                 for k, v in positions.items()}  # nb times played(csv)
    mmr_data = get_mmr_gain(data)  # mean and total mmr per champ
    n_games = len(data)  # total games(csv)
    tot_pickrate_n = {k: v/n_games for k,
                      v in nb_played.items()}  # total pickrate csv
    # pickrate new ( but not csv)
    pickrate_new = {k: nb_played[k]/v[0]
                    for k, v in picks_stats.items() if k in nb_played}
    nb_proposed_new = {k: v[0] for k, v in picks_stats.items()}
    percent_proposed_new = {k: v[0]/n_games for k, v in picks_stats.items()}
    top_n = get_top_n_rate(positions)
    all_heros = pickrate_new.keys()
    results = defaultdict(list)
    for hero in all_heros:
        if hero not in mean_pos:
            mean_pos[hero] = np.nan
        if hero not in mmr_data:
            mmr_data[hero] = [np.nan, np.nan, np.nan, np.nan]
        if hero not in nb_played:
            nb_played[hero] = np.nan
        if hero not in tot_pickrate_n:
            tot_pickrate_n[hero] = np.nan
        if hero not in nb_proposed_new:
            nb_proposed_new[hero] = np.nan
        if hero not in pickrate_new:
            pickrate_new[hero] = np.nan
        if hero not in percent_proposed_new:
            percent_proposed_new[hero] = 0
        results[hero] = [hero, round_(mean_pos[hero], 2),
                         round_(mmr_data[hero][0], 0),
                         mmr_data[hero][3], mmr_data[hero][2],
                         mmr_data[hero][1], nb_played[hero],
                         nb_proposed_new[hero], round_(
                             pickrate_new[hero]*100, 1),
                         round_(tot_pickrate_n[hero]*100, 1),
                         round_(percent_proposed_new[hero]*100, 1)
                         ]
    df = pd.DataFrame.from_dict(results, orient='index', columns=['nom', 'position moyenne',
                                                                  'mmr moyen par partie',
                                                                  'mmr total gagné',
                                                                  'mmr total perdu',
                                                                  'gain mmr',
                                                                  'nombre de pick',
                                                                  'nombre de fois proposé',
                                                                  'pickrate',
                                                                  '% de parties',
                                                                  '% proposé'
                                                                  ])

    top_n_temp = defaultdict(list)
    for hero, data_hero in top_n.items():
        hero = hero.replace('"', '')
        for place in range(1, 9):
            top_n_temp[hero].append(round_(data_hero[place]*100, 1))
        top_n_temp[hero].append(sum(top_n_temp[hero][:4]))
        top_n_temp[hero].insert(0, hero.replace('"', ''))
    df_top_n = pd.DataFrame.from_dict(top_n_temp, orient='index', columns=[
                                      'nom']+[f'% top {i}' for i in range(1, 9)]+['winrate'])
    df_all = pd.concat([df, df_top_n.drop('nom', axis=1)], axis=1)
    return df, df_top_n, df_all, all_matches_per_champ, mmr_evo, mean_position


def round_(nb, n=2):
    if nb == np.nan:
        return nb
    return round(nb, n)


def get_all_matches_per_champ(data):
    results = defaultdict(list)
    for index_match, match in enumerate(data):
        hero = match['hero']
        results[hero].append([match['date'] if len(match['date']) > 2 else np.nan,
                              match['pos'],
                              data[index_match -
                                   1]['mmr'] if index_match != 0 else data[0]['mmr'],
                              match['mmr'],
                              int(match['mmr'].replace('"', '')) - int(data[index_match-1]['mmr'].replace('"', '')) if index_match != 0 else 0])
    for hero, hero_data in results.items():
        for index in range(len(hero_data)):
            if index == 0:
                results[hero][index].append(results[hero][0][-1])
            else:
                results[hero][index].append(
                    results[hero][index-1][-1]+results[hero][index][-1])

    all_dfs = {k.replace('"', ''): pd.DataFrame(v, columns=[
        'date', 'position', 'mmr avant', 'mmr aprés', 'gain mmr', 'gain mmr total']) for k, v in results.items()}
    return all_dfs


def rolling_mean(data, w=3):
    results = []
    for index in range(len(data)):
        if index-w < 0:
            last_match = data[:index+1]
        else:
            last_match = data[index-w+1:index+1]
        results.append(np.mean(last_match))
    return results
