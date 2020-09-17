import pandas as pd
from collections import Counter, defaultdict
from config import LOG_PATH, CSV_PATH
try:
    from config import CSV_PATH_OLD
except:
    CSV_PATH_OLD = None
import numpy as np
from bg_logs_reader import parse_logs
import os

log2df = {'Aranna Starseeker': 'Aranna, Unleashed',
          'Yogg-Saron': "Yogg-Saron, Hope's End"}
to_del = ['Brann Bronzebeard']
df_tags = pd.read_csv(os.path.join(__file__.replace(
    '\compute_stats.py', ''), 'types.csv'))
df_tags = df_tags.fillna('')


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
    buggeds = []
    for match in log_data:
        lucks = []
        turns = [k for k in match.keys() if k in list(range(20))]
        if len(turns) == 0:
            continue
        last_turn = max(turns)
        last_board = match[last_turn]['board']
        combat_results = []
        boards = {}
        statss = {}
        for turn in range(1, last_turn+1):
            comp_type, stats, bugged = get_board_type_and_stats(
                match[turn]['board'])
            statss[turn] = stats
            buggeds.append(bugged)
            luck = True
            if len(match[turn]['winrates']) == 0:
                luck = False
            else:
                winrate = match[turn]['winrates'][0]
            if match[turn]['winner'] in ['None', 'Tie']:
                combat_result = 'None'
            else:
                if match[turn]['winner'] == match['hero']:
                    combat_result = 'win'
                    if luck:
                        lucks.append(100-winrate)
                else:
                    combat_result = 'loss'
                    if luck:
                        lucks.append(-winrate)
            combat_results.append(combat_result)
            boards[turn] = {'minions': match[turn]['board'],
                            'type': comp_type, 'result': combat_result, 'turn': turn}
        luck = np.mean(lucks)
        comp_type = get_comp_type(boards)
        results.append({'hero': match['hero'], 'pos': match['pos'],
                        'luck': luck, 'boards': boards,
                        'combat_results': combat_results,
                        'last_mmr': match['last_mmr'], 'mmr': match['new_mmr'],
                        'comp_type': comp_type,
                        'board_stats': statss})
    return results


def get_comp_type(boards):
    coeffs = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4,
              11: 5, 12: 5, 13: 6, 14: 6, 15: 7, 16: 7, 17: 8, 18: 8, 19: 9, 20: 9}
    all_types = np.unique([v['type'] for v in boards.values()])
    results = {t: 0 for t in all_types}
    for nb_turn, board_data in boards.items():
        results[board_data['type']] += coeffs[nb_turn]
    max_value = max(list(results.values()))
    comp_at_max = [k for k, v in results.items() if v == max_value]
    if len(comp_at_max) == 0:
        print(results)
        return None
    comp_type = comp_at_max[0]
    return comp_type


def get_board_type_and_stats(board):
    replace = {'goldrinn': 'goldrinn,  the great wolf',
               'tabbycat': 'alleycat',
               'murlocscout': 'murloc tidehunter',
               'spawnofnzoth':  "spawn of n'zoth",
               'murloc scout': 'murloc tidehunter',
               'deflectobot': 'deflect-o-bot',
               'kangorsapprentice': "kangor's apprentice",
               'mechanoegg': 'mechano-egg',
               'razorgoretheuntamed': 'razorgore, the untamed',
               'kalecgosarcaneaspect': 'kalecgos, arcane aspect',
               'goldrinnthegreatwolf': 'goldrinn,  the great wolf',
               'malganis': "mal'ganis",
               'oldmurkeye': 'old murk-eye',
               'bolvarfireblood': 'bolvar,  fireblood',
               'sneedsoldshredder': "sneed's old shredder",
               'razorgore': 'razorgore, the untamed',
               'nat pagle': 'nat pagle, extreme angler',
               'bolvar': 'bolvar,  fireblood',
               'kalecgos': 'kalecgos, arcane aspect',
               }
    old = {'zoobot': ['Mech'],
           'amalgam': ['Beast', 'Pirate', 'Dragon', 'Demon', 'Mech', 'Murloc'],
           'holymackerel': ['Murloc', 'Divine Shield'],
           'gentle megasaur': ['Murloc'],
           }
    if board == []:
        return 'None', None, []
    noms = np.array([n.lower().strip() for n in df_tags['Name'].values])
    noms_ = np.array([n.replace(' ', '') for n in noms])
    possible_tags = ['Beast', 'Pirate', 'Dragon', 'Demon',
                     'Mech', 'Murloc', 'Divine Shield', 'Deathrattle', 'Menagerie']
    board_tags = []
    bugged = []
    all_minions = {}
    tot_atq, tot_pv = 0, 0
    for minion in board:
        splitted = minion.split(',')
        atq = int(minion.split('(')[1].split(',')[0])
        try:
            pv = int(minion.split(',')[1].split(')')[0])
        except:
            pv = int(minion.split(',')[2].split(')')[0])
        tot_atq += atq
        tot_pv += pv
        if len(splitted) <= 1:
            continue
        name = splitted[0].split('(')[0].strip().lower()
        if name in replace:
            name = replace[name]
        all_minions[name] = {'atq': atq, 'pv': pv}
        idx = np.where(noms == name)[0]
        if len(idx) == 0:
            idx = np.where(noms_ == name)[0]
            if len(idx) == 0:
                if name in old:
                    tags = old[name]
                else:
                    bugged.append(name)
                    continue
            else:
                tags = df_tags.iloc[idx]['Combined'].values[0].split(',')
        else:
            tags = df_tags.iloc[idx]['Combined'].values[0].split(',')
        for tag in tags:
            if tag in possible_tags:
                board_tags.append(tag.strip())
    c_tags = Counter(board_tags)
    if 'pogo-hopper' in all_minions and all_minions['pogo-hopper']['pv'] >= 6:
        type_ = 'Pogo Hopper'
    elif 'lightfang enforcer' in all_minions:
        type_ = 'Menagerie'
    elif ('baron rivendare' in all_minions and c_tags['Deathrattle'] >= 2) or c_tags['Deathrattle'] >= 4:
        type_ = 'Deathrattle'
    elif (('bolvar,  fireblood' in all_minions and 'drakonid enforcer' in all_minions)
          or ('bolvar,  fireblood' in all_minions)
          or ('drakonid enforcer' in all_minions and (('Dragon' in c_tags and c_tags['Dragon'] <= 3) or 'Dragon' not in c_tags))):
        type_ = 'Divine Shield'
    else:
        races = {k: v for k, v in c_tags.items() if k in [
            'Beast', 'Pirate', 'Dragon', 'Demon', 'Mech', 'Murloc']}
        best_race, n = c_tags.most_common()[0]
        if len(races) >= 3:
            if n >= 3:
                type_ = best_race
            else:
                type_ = 'Menagerie'
        else:
            if len(c_tags.most_common()) > 1 and c_tags.most_common()[1][1] == n:
                type_ = 'Menagerie'
            else:
                type_ = best_race

    return type_, {'atq': tot_atq, 'pv': tot_pv}, bugged


def get_comp_types_stats(log_data):
    # stats selon le type de compo : % joué, winrate, cbt_winrate, net_mmr
    # loss mmr, win mmr, graph positions, mmr evo, idem que page champions
    all_types = Counter([l['comp_type'] for l in log_data])
    results = {}
    for type_, nb_played in all_types.items():
        subresults = {}
        all_type_data = [log for log in log_data if log['comp_type'] == type_]
        all_combat_results = [log['combat_results'] for log in all_type_data]
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


def get_cbt_winrate_per_comp(log_data):
    all_comps = ['Beast', 'Pirate', 'Dragon', 'Demon',
                 'Mech', 'Murloc', 'Divine Shield', 'Deathrattle', 'Pogo Hopper', 'Menagerie']
    results = {champ: [] for champ in all_comps}
    for comp in all_comps:
        all_comp_data = [d for a in [list(k.values()) for k in [
            l['boards'] for l in log_data]]for d in a if d['type'] == comp]
        all_turns = np.unique([d['turn'] for d in all_comp_data])
        max_turn = max(all_turns)
        for turn in range(1, max_turn):
            if turn not in all_turns:
                results[comp].append(50)
            all_turn_data = Counter([d['result']
                                     for d in all_comp_data if d['turn'] == turn])
            results[comp].append(
                round_(all_turn_data['win']/(all_turn_data['loss']+all_turn_data['win']+0.01)*100, 2))
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


def reorder_logs(log_data, df_data):
    results = []
    log_data = [l for l in log_data if l['comp_type'] != 'None']
    for match in df_data:
        hero = match['hero']
        mmr = int(match['mmr'])
        log_match = [l for l in log_data if l['hero']
                     == hero and int(l['mmr']) == mmr]
        if len(log_match) > 0:
            log_match[0]['date'] = match['date']
            results.append(log_match[0])
    return results


def get_board_popularity(log_data):
    all_comps = ['Beast', 'Pirate', 'Dragon', 'Demon',
                 'Mech', 'Murloc', 'Divine Shield', 'Deathrattle', 'Pogo Hopper', 'Menagerie']
    temp = defaultdict(list)
    for match in log_data:
        for turn, board in match['boards'].items():
            if board['type'] != 'None':
                temp[turn].append(board['type'])
    res = {k: Counter(v) for k, v in temp.items()}
    to_return = {}
    for comp_type in all_comps:
        comp_res = []
        for turn, turn_res in res.items():
            if comp_type in turn_res:
                comp_res.append(turn_res[comp_type]/sum(turn_res.values()))
            else:
                comp_res.append(0)
        to_return[comp_type] = comp_res
    return to_return


def build_df_board_stats(log_data):
    to_df = {}
    for index, data in enumerate(log_data):
        nb_turns = len(data['combat_results'])
        to_df[index] = [data['hero'], data['comp_type'], *
                        [sum(v.values()) for k, v in data['board_stats'].items() if v], *[np.nan for i in range(20-nb_turns)]]
    df = pd.DataFrame.from_dict(to_df, orient='index', columns=[
                                'hero', 'comp', *[i for i in range(1, 21)]])
    df.fillna(-1)
    return df


def build_df_games(log_data):
    to_df = {}
    for index, data in enumerate(log_data):
        to_df[index] = [data['date'], data['hero'],
                        data['comp_type'], data['pos']]
    df = pd.DataFrame.from_dict(to_df, orient='index', columns=[
                                'date', 'hero', 'comp', 'pos'])
    return df


def get_all_stats():
    # CSV data
    df_champ_new = pd.read_csv(CSV_PATH, sep=';').values
    df_champ = pd.read_csv(
        CSV_PATH_OLD, sep=';').values if CSV_PATH_OLD else None
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
    pickrate, nb_proposed = get_pickrate(log_data)
    percent_proposed = {k: v/len(log_data) for k, v in nb_proposed.items()}
    log_data = process_log_data(log_data)
    board_pop = get_board_popularity(log_data)
    cbt_winrate_champs = get_cbt_winrate_per_champ(log_data)
    cbt_winrate_comps = get_cbt_winrate_per_comp(log_data)
    df_board_stats = build_df_board_stats(log_data)
    log_data = reorder_logs(log_data, data)
    df_games = build_df_games(log_data)
    comp_types = get_comp_types_stats(log_data)
    comp_types_per_champ = get_comp_types_per_champ(log_data)
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
    df_types = pd.DataFrame.from_dict(type_results, orient='index', columns=['nom', 'position moyenne', 'winrate', 'mmr moyen par partie', 'gain mmr',
                                                                             'mmr total gagné', 'mmr total perdu', 'nombre de pick', *[f'% top {i}' for i in range(1, 9)], 'Victoires', 'Top 1'])

    return (df_champ, df_top_n_champ, df_all_champ, df_types,
            all_matches_per_champ, mmr_evo, mean_position,
            cbt_winrate_champs, cbt_winrate_comps, comp_types_per_champ,
            comp_types, df_board_stats, board_pop, df_games, log_data)


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
