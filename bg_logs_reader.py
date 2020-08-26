# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:52:36 2020

@author: simed
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
from collections import defaultdict
import numpy as np
from config import LOG_PATH, CSV_PATH


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
    mmr = [int(match_data['mmr'])
           for match_data in data if int(match_data['mmr']) > 3000]
    return mmr


def build_lists(path):
    with open(path, 'r') as f:
        data = f.read()

    lines = data.split('\n')
    hours = []
    types = []
    messages = []

    for line in lines:
        try:
            hour, type_, mess = line.split('|')
        except:
            continue
        hours.append(hour)
        types.append(type_)
        messages.append(mess)
    return hours, types, messages


def locate_starts(messages):
    results = []
    for index, mess in enumerate(messages):
        if 'Game start' in mess:
            results.append(index)
    return results


def locate_turns(messages):
    results = []
    for index, mess in enumerate(messages):
        if 'Player turn' in mess:
            results.append(index)
    return results


def extract_data(hours, messages):
    starts = locate_starts(messages)
    nb_matchs = len(starts)
    results = []
    for i, index_start in enumerate(starts):
        if i == len(starts)-1:
            messages_data = messages[index_start:]
            hours_data = hours[index_start:]
        else:
            messages_data = messages[index_start:starts[i+1]]
            hours_data = hours[index_start:starts[i+1]]
        results.append(extract_match_data(hours_data, messages_data))
    return results


def extract_match_data(hours, messages):
    turns = locate_turns(messages)
    turns.insert(0, 0)
    results = {'lvl': []}
    last_minions = []
    for i, index_turn in enumerate(turns):
        if i == len(turns)-1:
            messages_data = messages[index_turn:]
        else:
            messages_data = messages[index_turn:turns[i+1]]
        if i == 0:
            game_info = extract_game_info(messages_data)
            results['infos'] = game_info
        else:
            turn_info, minions = extract_turn_info(messages_data)
            last_minions = minions if len(minions) > 1 else last_minions
            results['lvl'].append(turn_info)
    new, prev, position = -1, -1, -1
    for index, line in enumerate(messages_data):
        if 'Game ended' in line:
            position = line[-1]
        if 'Prev Rating' in line:
            prev = line.split(':')[1]
        if 'Rating Updated' in line:
            new = line.split(':')[1]
    results['result'] = [position, prev, new]
    results['minions'] = last_minions
    return results


def extract_game_info(messages):
    results = {'choices': []}
    for i, line in enumerate(messages):
        if 'Player.CreateInHand' in line:
            hero = line.split(',')[2].split('=')[1]
            if 'NOOOOOO' in hero:
                hero = 'Illidan Stormrage'
            results['choices'].append(hero)
        if 'Player.Play' in line:
            hero = line.split(',')[2].split('=')[1]
            if 'NOOOOOO' in hero:
                hero = 'Illidan Stormrage'
            results['hero'] = hero
            break
    return results


def extract_turn_info(messages):
    line = messages[1]
    lvl = line[-1]
    try:
        int(lvl)
    except:
        lvl = -1
    minions = []
    for line in messages:
        if 'Current minions in play' in line:
            minions = line.split(':')[1].split(',')
    return int(lvl), minions


def get_all_position(data):
    results = defaultdict(list)
    for match in data:
        try:
            results[match['hero']].append(int(match['pos']))
        except:
            pass
    return results


def extract_hdt_logs(dirpath):
    all_new = []
    for path in os.listdir(dirpath):
        if path.endswith('txt'):
            p = f'{dirpath}\\{path}'
            hours, types, messages = build_lists(p)
            all_new.append(extract_data(hours, messages))
    return all_new


def extract_choices_and_pick(log_path):
    all_new = extract_hdt_logs(log_path)
    new = {'choice': [a['infos']['choices'] for d in all_new for a in d if a['infos']['choices'] != [] and 'The Coin' not in a['infos']['choices']],
           'pick':  [a['infos']['hero'] for d in all_new for a in d if a['infos']['choices'] != [] and 'The Coin' not in a['infos']['choices']]}

    return new


def get_mmr_gain(data):
    results = defaultdict(list)
    for index_match, result_match in enumerate(data):
        if index_match == 0:
            results[result_match['hero']].append(0)
        else:
            results[result_match['hero']].append(
                int(result_match['mmr']) - int(data[index_match-1]['mmr']))
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
                              int(match['mmr']) - int(data[index_match-1]['mmr']) if index_match != 0 else 0])
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
# %%
