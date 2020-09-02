import pandas as pd
from collections import Counter, defaultdict
from config import LOG_PATH, CSV_PATH, CSV_PATH_OLD
import numpy as np
from bg_logs_reader import parse_logs
import os

log2df = {'Aranna Starseeker': 'Aranna, Unleashed',
          'Yogg-Saron': "Yogg-Saron, Hope's End"}
to_del = ['Brann Bronzebeard']


def df_to_dict(df, df_new):
    """
    transform the dataframe in a dict
    we cannot use df.to_dict because the csv have different format in it
    """
    dics = []
    if df is not None:
        for index, match_data in enumerate(df):
            elements = match_data
            i = 1 if elements[1] in['"Yogg-Saron',
                                    '"Aranna'] else 0  # names containing ','
            hero = elements[1]
            if hero == '"Aranna':
                hero = "Aranna, Unleashed"
            elif hero == '"Yogg-Saron':
                hero = "Yogg-Saron, Hope's End"
            else:
                hero = elements[1]
            dics.append({'date': elements[0],
                         'hero': hero.replace('"', ''),
                         'pos': str(elements[2+i]).replace('"', ''),
                         'mmr': str(elements[3+i]).replace('"', '')
                         })
    if df_new is not None:
        for index, match_data in enumerate(df_new):
            elements = match_data[0].split(',')
            i = 1 if elements[1] in['"Yogg-Saron',
                                    '"Aranna'] else 0  # names containing ','
            hero = elements[1]
            if hero == '"Aranna':
                hero = "Aranna, Unleashed"
            elif hero == '"Yogg-Saron':
                hero = "Yogg-Saron, Hope's End"
            dics.append({'date': elements[0],
                         'hero': hero.replace('"', ''),
                         'pos': str(elements[2+i]).replace('"', ''),
                         'mmr': str(elements[3+i]).replace('"', '')
                         })
    return dics


def get_all_matches_per_champ(data):
    results = defaultdict(list)
    for index_match, match in enumerate(data):
        hero = match['hero']
        results[hero].append([match['date'],
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
            if hero is not 'None':
                choices.append(hero)
        picks.append(match['hero'])
    choices, picks = Counter(choices), Counter(picks)
    results = {k: picks[k]/v for k, v in choices.items()}
    return results, choices


def process_log_data(log_data):
    """
    This function processes the log data in order to get a better view for computing
    battle luck, comp type and combat winrate
    """
    results = []
    for match in log_data:
        lucks = []
        turns = [k for k in match.keys() if type(k) == int]
        if len(turns) == 0:
            continue
        last_turn = max([k for k in match.keys() if type(k) == int])
        last_board = match[last_turn]['board']
        combat_result = []
        for i in range(1, last_turn+1):
            luck = True
            if len(match[i]['winrates']) == 0:
                luck = False
            else:
                winrate = match[i]['winrates'][0]
            if match[i]['winner'] in ['None', 'Tie']:
                combat_result.append('None')
            else:
                if match[i]['winner'] == match['hero']:
                    combat_result.append('win')
                    if luck:
                        lucks.append(100-winrate)
                else:
                    combat_result.append('loss')
                    if luck:
                        lucks.append(-winrate)
        luck = np.mean(lucks)
        comp_type = get_board_type(last_board)
        results.append({'hero': match['hero'], 'pos': match['pos'],
                        'luck': luck, 'board': last_board,
                        'combat_results': combat_result,
                        'last_mmr': match['last_mmr'], 'mmr': match['new_mmr'],
                        'comp_type': comp_type})
    return results


def get_board_type(board):
    if board == []:
        return 'None'
    df_tags = pd.read_csv(os.path.join(__file__.replace(
        '\compute_stats.py', ''), 'types.csv'), sep=';')
    noms = df_tags['nom'].values
    possible_tags = {'d': 'divin',
                     'r': 'death', }
    possible = ['pirate', 'mech', 'murloc', 'dragon',
                'big demon', 'token demon', 'beast', 'death', 'menagerie', 'divin']
    board_tags = []
    for minion in board:
        splitted = minion.split(',')
        if len(splitted) <= 1:
            continue
        tags = splitted[2:]
        for tag in tags:
            if tag in possible_tags:
                board_tags.append(possible_tags[tag])
        name = splitted[0].split('(')[0]
        df_data = df_tags.iloc[np.where(noms == name)[0]]
        tags_idx = np.where(df_data.any().values[1:])[0]
        for tag_idx in tags_idx:
            board_tags.append(list(df_data.columns)[tag_idx+1])
    c_tags = Counter(board_tags)
    winner = c_tags.most_common()[0][0]
    return winner


def get_comp_types_stats(log_data):
    # stats selon le type de compo : % joué, winrate, cbt_winrate, net_mmr
    # loss mmr, win mmr, graph positions, mmr evo, idem que page champions
    all_types = Counter([l['comp_type'] for l in log_data])
    results = {}
    for type_, nb_played in all_types.items():
        subresults = {}
        all_type_data = [log for log in log_data if log['comp_type'] == type_]
        all_combat_results = [log['combat_results'] for log in all_type_data]
        subresults['cbt_winrate'] = get_cbt_winrate(all_combat_results)
        subresults['%_joué'] = nb_played/len(log_data)
        subresults['nombre de fois joué'] = nb_played
        all_positions = Counter([log['pos']
                                 for log in all_type_data if log['pos'] != -1])
        subresults['placement moyen'] = np.mean(
            [int(log['pos']) for log in all_type_data if log['pos'] != -1])
        top_n = {k: v/sum(a for a in all_positions.values())
                 for k, v in all_positions.items()}
        for k, v in top_n.items():
            subresults[f'% top {k}'] = round_(
                v)
        for i in range(1, 9):
            if f'% top {i}' not in subresults:
                subresults[f'% top {i}'] = 0
        subresults['winrate'] = round_(
            sum([v for k, v in top_n.items() if int(k) <= 4]))
        mmr_evo = [int(l['mmr'])-int(l['last_mmr']) for l in all_type_data]
        mmr_evo = [mmr for mmr in mmr_evo if mmr < 120 and mmr > -120]
        subresults['net mmr'] = sum(mmr_evo)
        subresults['gain mmr absolu'] = sum([m for m in mmr_evo if m > 0])
        subresults['perte mmr absolu'] = sum([-m for m in mmr_evo if m < 0])
        subresults['mean mmr'] = sum(mmr_evo)/nb_played
        subresults['mmr_evo'] = [mmr_evo[i] +
                                 sum(mmr_evo[:i]) for i in range(len(mmr_evo))]
        subresults['Nb victoire'] = round_(
            nb_played * subresults['winrate'], 0)
        subresults['Nb top 1'] = round_(nb_played * subresults['% top 1'], 0)
        subresults['champs_stats'] = get_champ_stats_per_type(all_type_data)
        results[type_] = subresults
    return results


def get_champ_stats_per_type(data):
    res = {}
    all_champs = [l['hero'] for l in data]
    for champ in all_champs:
        if champ == '':
            continue
        champ_data = [l for l in data if l['hero'] == champ]
        mmrs = [int(l['mmr'])-int(l['last_mmr']) for l in champ_data]
        mmrs = [mmr for mmr in mmrs if mmr < 120 and mmr > -120]
        pos = [l['pos'] for l in champ_data]
        res[champ] = {'mmr': np.sum(mmrs), 'pos': pos,
                      'nb_played': len(champ_data)}
    return res


def get_cbt_winrate_per_champ(log_data):
    # cbt winrate par champion
    all_champs = list(Counter([log['hero'] for log in log_data]).keys())
    results = {champ: [0]*20 for champ in all_champs if champ != ''}
    for champ in all_champs:
        if champ == '':
            continue
        all_champ_data = [log for log in log_data if log['hero'] == champ]
        all_combat_results = [log['combat_results'] for log in all_champ_data]
        results[champ] = get_cbt_winrate(all_combat_results)
    return results


def get_cbt_winrate(data):
    champ_winrates = []
    turn_max = max([len(res) for res in data])
    for i in range(turn_max):
        turn_results = Counter([res[i] for res in data if len(res) > i])
        if 'win' not in turn_results:
            champ_winrates.append(0)
        elif 'loss' not in turn_results:
            champ_winrates.append(100)
        else:
            champ_winrates.append(
                100*turn_results['win']/(turn_results['loss']+turn_results['win']))
    if max(champ_winrates) == 0:
        return champ_winrates
    while champ_winrates[-1] == 0:
        del champ_winrates[-1]
    return champ_winrates


def get_comp_types_per_champ(log_data):
    # Position moyenne par type de compo, par champion
    # pourcentage de fois joué chaque type de compo, par perso
    all_champs = Counter([l['hero'] for l in log_data])
    results = {}
    for champ in all_champs.keys():
        net_mmrs = {}
        positions = {}
        if champ == '':
            continue
        all_champ_data = [l for l in log_data if l['hero'] == champ]
        all_types_played = Counter([l['comp_type'] for l in all_champ_data])
        for type_, nb_played in all_types_played.items():
            type_data = [l for l in all_champ_data if l['comp_type'] == type_]
            pos = [l['pos'] for l in type_data]
            mmr_evo = [int(l['mmr'])-int(l['last_mmr']) for l in type_data]
            mmr_evo = [mmr for mmr in mmr_evo if mmr < 120 and mmr > -120]
            positions[type_] = pos
            net_mmrs[type_] = np.sum(mmr_evo)
        results[champ] = {'net_mmr': net_mmrs,
                          'types': all_types_played, 'position': positions}
    return results


def match_log_to_df(log_data, df_data):
    results = []
    for log in log_data:
        hero = log['hero']
        mmr = int(log['new_mmr'])
        matched = [d for d in df_data if d['hero']
                   == hero and int(d['mmr']) == mmr]
        if len(matched) > 0:
            results.append([log, matched[0]])
    return results


all_dfs = [d[1] for d in results]
not_matched = [d for d in df_data if d not in all_dfs]


def get_all_stats():
    # CSV data
    df_champ_new = pd.read_csv(CSV_PATH, sep=';').values
    df_champ = pd.read_csv(CSV_PATH_OLD, sep=';').values
    data = df_to_dict(df_champ, df_champ_new)
    all_matches_per_champ = get_all_matches_per_champ(data)
    all_heros = list(all_matches_per_champ.keys())
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
    # battle luck,
    pickrate, nb_proposed = get_pickrate(log_data)
    percent_proposed = {k: v/len(log_data) for k, v in nb_proposed.items()}
    log_data = process_log_data(log_data)
    cbt_winrate_champs = get_cbt_winrate_per_champ(log_data)
    data_wo_none = [l for l in log_data if l['comp_type'] != 'None']
    comp_types = get_comp_types_stats(data_wo_none)
    comp_types_per_champ = get_comp_types_per_champ(data_wo_none)
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
        if hero not in nb_proposed:
            nb_proposed[hero] = np.nan
        if hero not in pickrate:
            pickrate[hero] = np.nan
        if hero not in percent_proposed:
            percent_proposed[hero] = 0
        results[hero] = [hero, round_(champ_mean_pos[hero], 2),
                         round_(champ_mmr_data[hero][0], 0),
                         champ_mmr_data[hero][3], champ_mmr_data[hero][2],
                         champ_mmr_data[hero][1], nb_played[hero],
                         nb_proposed[hero], round_(
                             pickrate[hero]*100, 1),
                         round_(champ_played_percentage[hero]*100, 1),
                         round_(percent_proposed[hero]*100, 1)
                         ]
    df_champ = pd.DataFrame.from_dict(results, orient='index', columns=['nom', 'position moyenne',
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
    df_top_n_champ = pd.DataFrame.from_dict(top_n_temp, orient='index', columns=[
        'nom']+[f'% top {i}' for i in range(1, 9)]+['winrate'])
    df_all_champ = pd.concat(
        [df_champ, df_top_n_champ.drop('nom', axis=1)], axis=1)
    battle_luck = [[l['luck'], int(
        l['pos'])] for l in log_data if l['luck'] != np.nan and l['pos'] != -1]
    type_results = {}
    for type_, type_data in comp_types.items():
        type_results[type_] = [type_, round_(type_data['placement moyen'], 2), type_data['winrate']*100, round_(type_data['mean mmr'], 1),
                               type_data['net mmr'], type_data['gain mmr absolu'], type_data['perte mmr absolu'],
                               type_data['nombre de fois joué'], *[type_data[f'% top {i}'] for i in range(1, 9)], type_data['Nb victoire'], type_data['Nb top 1']]
    df_types = pd.DataFrame.from_dict(type_results, orient='index', columns=['nom', 'position moyenne', 'winrate', 'mmr moyen par partie', 'net mmr',
                                                                             'mmr total gagné', 'mmr total perdue', 'nombre de pick', *[f'% top {i}' for i in range(1, 9)], 'Victoires', 'Top 1'])

    return (df_champ, df_top_n_champ, df_all_champ, df_types,
            all_matches_per_champ, mmr_evo, mean_position,
            cbt_winrate_champs, comp_types_per_champ, comp_types, battle_luck)


# def get_all_stats_():
#     choices_and_pick = extract_choices_and_pick(LOG_PATH)
#     picks_stats = get_pick_stats(choices_and_pick)
#     df = pd.read_csv(CSV_PATH, sep=';').values
#     data = df_to_dict(df)
#     all_matches_per_champ = get_all_matches_per_champ(data)
#     mmr_evo = get_all_mmr(data)
#     champs_pos = get_all_position(data)
#     mean_position = round_(
#         np.mean([v for x in champs_pos.values() for v in x]), 2)  # global mean pos
#     champ_mean_pos = {k.replace('"', ''): np.mean(
#         v) for k, v in champs_pos.items() if len(v) != 0}  # champ mean positions
#     nb_played = {k.replace('"', ''): len(v)
#                  for k, v in champs_pos.items()}  # nb times played(csv)
#     champ_mmr_data = get_mmr_gain(data)  # mean and total mmr per champ
#     n_games = len(data)  # total games(csv)
#     champ_played_percentage = {k: v/n_games for k,
#                                v in nb_played.items()}  # played percentage
#     # pickrate new ( but not csv)
#     pickrate_new = {k: nb_played[k]/v[0]
#                     for k, v in picks_stats.items() if k in nb_played}
#     nb_proposed_new = {k: v[0] for k, v in picks_stats.items()}
#     percent_proposed_new = {k: v[0]/n_games for k, v in picks_stats.items()}
#     top_n = get_top_n_rate(champs_pos)
#     all_heros = pickrate_new.keys()
#     results = defaultdict(list)
#     for hero in all_heros:
#         if hero not in champ_mean_pos:
#             champ_mean_pos[hero] = np.nan
#         if hero not in champ_mmr_data:
#             champ_mmr_data[hero] = [np.nan, np.nan, np.nan, np.nan]
#         if hero not in nb_played:
#             nb_played[hero] = np.nan
#         if hero not in champ_played_percentage:
#             champ_played_percentage[hero] = np.nan
#         if hero not in nb_proposed_new:
#             nb_proposed_new[hero] = np.nan
#         if hero not in pickrate_new:
#             pickrate_new[hero] = np.nan
#         if hero not in percent_proposed_new:
#             percent_proposed_new[hero] = 0
#         results[hero] = [hero, round_(champ_mean_pos[hero], 2),
#                          round_(champ_mmr_data[hero][0], 0),
#                          champ_mmr_data[hero][3], champ_mmr_data[hero][2],
#                          champ_mmr_data[hero][1], nb_played[hero],
#                          nb_proposed_new[hero], round_(
#                              pickrate_new[hero]*100, 1),
#                          round_(champ_played_percentage[hero]*100, 1),
#                          round_(percent_proposed_new[hero]*100, 1)
#                          ]
#     df = pd.DataFrame.from_dict(results, orient='index', columns=['nom', 'position moyenne',
#                                                                   'mmr moyen par partie',
#                                                                   'mmr total gagné',
#                                                                   'mmr total perdu',
#                                                                   'gain mmr',
#                                                                   'nombre de pick',
#                                                                   'nombre de fois proposé',
#                                                                   'pickrate',
#                                                                   '% de parties',
#                                                                   '% proposé'
#                                                                   ])

#     top_n_temp = defaultdict(list)
#     for hero, data_hero in top_n.items():
#         hero = hero.replace('"', '')
#         for place in range(1, 9):
#             top_n_temp[hero].append(round_(data_hero[place]*100, 1))
#         top_n_temp[hero].append(sum(top_n_temp[hero][:4]))
#         top_n_temp[hero].insert(0, hero.replace('"', ''))
#     df_top_n = pd.DataFrame.from_dict(top_n_temp, orient='index', columns=[
#                                       'nom']+[f'% top {i}' for i in range(1, 9)]+['winrate'])
#     df_all = pd.concat([df, df_top_n.drop('nom', axis=1)], axis=1)
#     return df, df_top_n, df_all, all_matches_per_champ, mmr_evo, mean_position


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
