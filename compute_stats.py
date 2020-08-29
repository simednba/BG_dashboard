import pandas as pd
from collections import Counter, defaultdict
from config import LOG_PATH, CSV_PATH
import numpy as np
from bg_logs_reader import parse_logs


def df_to_dict(df):
    """
    transform the dataframe in a dict
    we cannot use df.to_dict because the csv have different format in it
    """
    dics = []
    for index, match_data in enumerate(df):
        elements = match_data[0].split(',')
        i = 1 if elements[1] in['"Yogg-Saron',
                                '"Aranna'] else 0  # names containing ','
        hero = elements[1] if elements[1] != '"Aranna' else "Aranna, Unleashed"
        dics.append({'date': elements[0],
                     'hero': hero.replace('"', ''),
                     'pos': elements[2+i].replace('"', ''),
                     'mmr': elements[3+i].replace('"', '')
                     })
    return dics


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


def get_all_mmr(data):
    mmr = [int(match_data['mmr'].replace('"', ''))
           for match_data in data if int(match_data['mmr'].replace('"', '')) > 3000]
    return mmr


def get_all_position(data):
    results = defaultdict(list)
    for match in data:
        results[match['hero']].append(int(match['pos']))
    return results


def get_mmr_gain(data):
    results = defaultdict(list)
    for index_match, result_match in enumerate(data):
        if index_match == 0:
            results[result_match['hero']].append(0)
        else:
            results[result_match['hero']].append(
                int(result_match['mmr'].replace('"', '')) - int(data[index_match-1]['mmr'].replace('"', '')))
    return {k.replace('"', ''): (np.mean(v), sum(v), -sum([i for i in v if i < 0]), sum([i for i in v if i > 0])) for k, v in results.items()}


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


def get_pickrate(log_data):
    choices = []
    picks = []
    for match in log_data:
        for hero in match['choices']:
            choices.append(hero)
        picks.append(match['hero'])
    choices, picks = Counter(choices), Counter(picks)
    results = {k: picks[k]/v for k, v in choices.items()}
    return results, choices

# TODO : bug sur la moitié des games & changer board pour board type


def process_log_data(log_data):
    """
    This function processes the log data in order to get a better view for computing
    battle luck, comp type and combat winrate
    """
    results = []
    for match in log_data:
        lucks = []
        try:
            last_turn = max([k for k in match.keys() if type(k) == int])
            last_board = match[last_turn]['board']
            combat_result = []
            for i in range(1, last_turn+1):
                winrate = match[i]['winrates'][0]
                if match[i]['winner'] in ['None', 'Tie']:
                    combat_result.append('None')
                else:
                    if match[i]['winner'] == match['hero']:
                        combat_result.append('win')
                        lucks.append(100-winrate)
                    else:
                        combat_result.append('loss')
                        lucks.append(-winrate)
            luck = np.mean(lucks)
            results.append({'hero': match['hero'], 'pos': match['pos'],
                            'luck': luck, 'board': last_board,
                            'combat_results': combat_result})
        except:  # some data are weird, just 2 games
            print(match)
            continue
    return results


def get_board_type(board):
    pass


def get_comp_types_stats(log_data):
    # stats selon le type de compo : % joué, winrate, cbt_winrate, net_mmr
    # loss mmr, win mmr, graph positions, mmr evo, idem que page champions
    pass


def get_cbt_winrate_per_champ(log_data):
    # cbt winrate par champion
    results = {}
    for match in log_data:
        hero = match['hero']
        if hero not in results:
            subresults = {i: [] for i in range(20)}
        else:
            subresults = results[hero]
        for index, result in enumerate(match['combat_results']):
            subresults[index].append(result)
        results[hero] = subresults
    final_results = {}
    for hero, res in results.items():
        sub = {k: len(np.where(np.array()))}


def get_comp_types_per_champ(log_data):
    # Position moyenne par type de compo, par champion
    # pourcentage de fois joué chaque type de compo, par perso
    pass


def get_all_stats():
    # CSV data
    df = pd.read_csv(CSV_PATH, sep=';').values
    data = df_to_dict(df)
    all_matches_per_champ = get_all_matches_per_champ(data)
    mmr_evo = get_all_mmr(data)
    champs_pos = get_all_position(data)
    mean_position = round_(
        np.mean([v for x in champs_pos.values() for v in x]), 2)  # global mean pos
    champ_mean_pos = {k.replace('"', ''): np.mean(
        v) for k, v in champs_pos.items() if len(v) != 0}  # champ mean positions
    nb_played = {k.replace('"', ''): len(v)
                 for k, v in champs_pos.items()}  # nb times played(csv)
    mmr_data = get_mmr_gain(data)  # (mean, net, lost, won)
    champ_mmr_data = get_mmr_gain(data)  # mean and total mmr per champ
    n_games = len(data)  # total games(csv)
    champ_played_percentage = {k: v/n_games for k,
                               v in nb_played.items()}  # played percentage
    top_n = get_top_n_rate(champs_pos)
    # LOG data
    log_data = [data for a in parse_logs(LOG_PATH) for data in a]
    # pickrate, last_board& pos, battle luck,
    pickrate, nb_proposed = get_pickrate(log_data)
    percent_proposed = {k: v/len(log_data) for k, v in nb_proposed.items()}
    log_data_pretty = process_log_data(log_data)


def get_all_stats_():
    choices_and_pick = extract_choices_and_pick(LOG_PATH)
    picks_stats = get_pick_stats(choices_and_pick)
    df = pd.read_csv(CSV_PATH, sep=';').values
    data = df_to_dict(df)
    all_matches_per_champ = get_all_matches_per_champ(data)
    mmr_evo = get_all_mmr(data)
    champs_pos = get_all_position(data)
    mean_position = round_(
        np.mean([v for x in champs_pos.values() for v in x]), 2)  # global mean pos
    champ_mean_pos = {k.replace('"', ''): np.mean(
        v) for k, v in champs_pos.items() if len(v) != 0}  # champ mean positions
    nb_played = {k.replace('"', ''): len(v)
                 for k, v in champs_pos.items()}  # nb times played(csv)
    champ_mmr_data = get_mmr_gain(data)  # mean and total mmr per champ
    n_games = len(data)  # total games(csv)
    champ_played_percentage = {k: v/n_games for k,
                               v in nb_played.items()}  # played percentage
    # pickrate new ( but not csv)
    pickrate_new = {k: nb_played[k]/v[0]
                    for k, v in picks_stats.items() if k in nb_played}
    nb_proposed_new = {k: v[0] for k, v in picks_stats.items()}
    percent_proposed_new = {k: v[0]/n_games for k, v in picks_stats.items()}
    top_n = get_top_n_rate(champs_pos)
    all_heros = pickrate_new.keys()
    results = defaultdict(list)
    for hero in all_heros:
        if hero not in champ_mean_pos:
            champ_mean_pos[hero] = np.nan
        if hero not in champ_mmr_data:
            champ_mmr_data[hero] = [np.nan, np.nan, np.nan, np.nan]
        if hero not in nb_played:
            nb_played[hero] = np.nan
        if hero not in champ_played_percentage:
            champ_played_percentage[hero] = np.nan
        if hero not in nb_proposed_new:
            nb_proposed_new[hero] = np.nan
        if hero not in pickrate_new:
            pickrate_new[hero] = np.nan
        if hero not in percent_proposed_new:
            percent_proposed_new[hero] = 0
        results[hero] = [hero, round_(champ_mean_pos[hero], 2),
                         round_(champ_mmr_data[hero][0], 0),
                         champ_mmr_data[hero][3], champ_mmr_data[hero][2],
                         champ_mmr_data[hero][1], nb_played[hero],
                         nb_proposed_new[hero], round_(
                             pickrate_new[hero]*100, 1),
                         round_(champ_played_percentage[hero]*100, 1),
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


def rolling_mean(data, w=3):
    results = []
    for index in range(len(data)):
        if index-w < 0:
            last_match = data[:index+1]
        else:
            last_match = data[index-w+1:index+1]
        results.append(np.mean(last_match))
    return results
