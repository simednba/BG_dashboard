BG_dashboard

# Introduction

Welcome to this page that explains how BG_dashboard works, how to install it, and what you can do with it.
BG_dahboard is a program that will allow you to display a lot of statistics about your gaming sessions on Hearthstone Battlegrounds, in a user-friendly way.
To do so, BG_dashboard will read the log files left on your computer to extract statistics on your gaming sessions. BG_dashboard will not directly read Hearthstone logs, but rather HearthstoneDeckTracker logs.

The consequence of this is that this program will not work without HearthstoneDeckTracker.
Another consequence is that this program is retro compatible : if you have been playing since the beginning with Hearthstone Deck Tracker activated, you will be able to have your statistics on these games after installing the program.

# Avalaible statistics
For the moment, here are the statistics which you will have access to: 
## General
- MMR over time, with or without rolling mean **(need JawsLouis plugins)**
- Placements repartition, showed as a pie or as a bar graph (similar to hsreplay)
- Compositions played (%)
- Composition popularity by turn

## Comparison between different heros or comps
- pickrate
- proposed rate
- total MMR won
- total MMR lost
- Net MMR
- Winrate
- top 1 rate

## for each hero:
- MMR over time **(need JawsLouis plugins)**
- Placement repartition
- combat winrate
- composition played
- net MMR & mean position per composition type
- last games

## for each composition type(Beast, Murloc ...)
- MMR over time **(need JawsLouis plugins)**
- Placement repartition
- combat winrate
- Hero played
- net MMR & mean position per hero


for some examples : [Link](https://www.notion.so/BG-images-1e837b7a40cd4e97b067c2bf96f399e2)



# Installation

## Requirements
- python 3.6 or later
- Hearthstone Deck Tracker : [Link](https://hsreplay.net/downloads/?hl=en)
- (Optionnal) JawsLouis plugin for Hearthstone deck tracker : [Link](https://github.com/jawslouis/Battlegrounds-Match-Data)
    ( Note that without this plugins, some statistics will be unavalaible)

## Installation process
- clone the repo with git : git clone git@github.com:simednba/BG_dashboard.git
- install dependencies :
    - cd BG_dashboard
    - pip install -r requirements.txt
- fill the config.py file with the requested information: path to the CSV generated by the JawsLouis plugin, path to the hearthstone deck tracker logs (usually C:\\Users\\username\\AppData\\Roaming\\HearthstoneDeckTracker\\BattlegroundsMatchData.csv and C:\\Users\\username\\AppData\\Roaming\\HearthstoneDeckTracker\\Logs)

# Usage
- open a command line and go in the script folder
- run in command line : python "batlleground dashboard.py"
- open a web browser, and go to **http://127.0.0.1:8050/**
- Enjoy

# Notes
 There are some rules to follow if you want the statistics collected by this program to be accurate: 
- always activate hearthstone deck tracker when you play (if a game is played without HDT, or if you play on a cell phone for example, this game will not be included in the stats)
- do not give up a game: in this case, the game will be excluded from the statistics
- when updating Hearthstone, do not play before Hearthstone Deck Tracker is updated (about 12 hours on average after Hearthstone update).
- when fininshing a game, if the CSV created by JawsLouis plugin is opened, then the game will not be recorded.
 


