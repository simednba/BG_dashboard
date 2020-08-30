"""
Created on Thu Mar 12 18:52:36 2020

@author: simed
"""
import os


def extract_logs(path):
    """
    Extract relevant messages from log file
    """
    with open(path, 'r') as f:
        data = f.read()
    lines = data.split('\n')
    messages = []
    for line in lines:
        try:
            _, _, mess = line.split('|')
        except:
            continue
        messages.append(mess)
    return messages


def locate_starts(messages):
    """
    locate the indexes of lines where a game start
    """
    results = []
    for index, mess in enumerate(messages):
        if 'Game start' in mess:
            results.append(index)
    return results


def in_bg_mode(match_data):
    """
    return True if the match data is for a valid battleground match
    """
    if len(match_data) < 200:
        return False
    if 'BgMatchData.InBgMode >> Game Start - Not in Battlegrounds Mode.' in match_data:
        return False
    return True


def extract_log_data(messages):
    """
    Extracts all matches data contained in messages.
    messages is a list of all log line from a single log file
    """
    starts_idx = locate_starts(messages)
    results = []
    for i, index_start in enumerate(starts_idx):
        if i == len(starts_idx)-1:
            matches_logs = messages[index_start:]  # Last match data
        else:
            matches_logs = messages[index_start:starts_idx[i+1]]
        if in_bg_mode(matches_logs):
            matches_data = extract_match_data(matches_logs)
            if matches_data is None:
                continue
            results.append(matches_data)
    return results


def get_player(match_data):
    """
    Get the name of the player of this match
    """
    player = 'None'
    for index, line in enumerate(match_data):
        if 'GameV2.UpdatePlayers' in line:
            return line.split('>>')[1].split('[')[0].split('#')[0].strip()
    return player


def locate_turns(messages):
    """
    locate turn starts indexes
    """
    results = []
    for index, mess in enumerate(messages):
        if 'Player turn' in mess:
            results.append(index)
    return results


def extract_game_choices_and_pick(messages):
    """
    Extracts a game choices and pick
    """
    results = {'choices': [], 'hero': ''}
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


def extract_board(simulation_data):
    """
    extract the player board
    """
    minions = []
    for line in simulation_data:
        minions.append(line.split('>>')[1])
    return minions


def extract_turn_info(turn_data):
    """
    Extract a turn information
    currently extracted : 
        - turn board via bobs buddy
        - combat winrate via bubsbuddy
        - combat result
    """
    start_sim, end_sim, sim_results, cbt_result = None, None, None, None
    for index, line in enumerate(turn_data):
        if 'BobsBuddyInvoker.RunSimulation >> ----- Simulation Input -----' in line:
            start_sim = index
        if 'BobsBuddyInvoker.RunSimulation >> Opponent' in line:
            end_sim = index
        if 'BobsBuddyInvoker.RunSimulation >> ----- Simulation Output -----' in line:
            sim_results = index+2
        if 'BobsBuddyInvoker.UpdateAttackingEntities' in line:
            cbt_result = index
    if start_sim and end_sim:
        simulation_data = turn_data[start_sim+2:end_sim]
        board = extract_board(simulation_data)
    else:
        board = []
    if sim_results:
        line = turn_data[sim_results].split('>>')[1].strip()
        winrates = [float(e.split('=')[1].replace('%', '').replace(
            ')', '').replace(',', '')) for e in line.split(' ')]

    else:
        winrates = []
    if cbt_result:
        winner = turn_data[cbt_result].split(
            '>>')[1].split(',')[0].split('=')[1]
    else:
        if start_sim:
            winner = 'Tie'
        else:
            winner = 'None'
    return board, winrates, winner


def extract_match_data(match_data):
    """
    Extract one match data
    """
    player = get_player(match_data)
    if player is 'None':
        return None
    turns = locate_turns(match_data)
    turns.insert(0, 0)
    results = {}
    for i, index_turn in enumerate(turns):
        if i == len(turns)-1:
            messages_data = match_data[index_turn:]
        else:
            messages_data = match_data[index_turn:turns[i+1]]
        if i == 0:
            game_info = extract_game_choices_and_pick(messages_data)
            results['choices'] = game_info['choices']
            results['hero'] = game_info['hero']
        else:
            board, winrates, winner = extract_turn_info(messages_data)
            results[i] = {'board': board,
                          'winrates': winrates, 'winner': winner}
    new, prev, position = -1, -1, -1
    for index, line in enumerate(messages_data):
        if 'Game ended' in line:
            position = line[-1]
        if 'Prev Rating' in line:
            prev = line.split(':')[1]
        if 'Rating Updated' in line:
            new = line.split(':')[1]
    results['pos'] = position
    results['player'] = player
    results['last_mmr'] = prev
    results['new_mmr'] = new
    return results


def parse_logs(logpath):
    """
    Parse the log files in logpath.
    returns a list with, for each element, informations in one log file
    """
    all_new = []
    for path in os.listdir(logpath):
        if path.endswith('txt'):
            p = f'{logpath}\\{path}'
            messages = extract_logs(p)
            all_new.append(extract_log_data(messages))
    return all_new
