# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:24:02 2020

@author: simed
"""

import os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.graph_objs as go
from compute_stats import get_all_stats, rolling_mean, round_
import dash_table
import numpy as np
from PIL import Image
from config import IMG_PATH


imgs = {}
for filename in os.listdir(IMG_PATH):
    if filename.lower().endswith('.png'):
        path = os.path.join(IMG_PATH, filename)
        imgs[filename[:-4]] = Image.open(path)

imgs_types = {}
for filename in os.listdir(IMG_PATH.replace('images', 'images_types')):
    if filename.lower().endswith('.png'):
        path = os.path.join(IMG_PATH.replace(
            'images', 'images_types'), filename)
        imgs_types[filename[:-4]] = Image.open(path)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

(df_stats_champs, df_top_champs, df_all_champ, df_types,  all_matches_champs,
 mmr, mean_position, cbt_winrate_champs, comp_types_per_champ,
 comp_types, battle_luck) = get_all_stats()
df_all_champ.loc['global', df_stats_champs.describe(
).columns] = df_stats_champs.describe().loc['mean'].round(2)
df_all_champ.loc['global', 'nom'] = 'moyenne'


app.layout = html.Div(children=[dcc.Tabs(id='main', value='main_v', children=[
    dcc.Tab(label='Global', value='global'),
    dcc.Tab(label='Toute les Données', value='all'),
    dcc.Tab(label='Par Perso', value='solo'),
    dcc.Tab(label='Par Compo', value='comps'),
    dcc.Tab(label='Comparaison', value='compar'),
]),
    html.Div(id='content'),


])

graphs_generals = {'MMR': 'mmr',
                   'MMR avec moyenne': 'mean_mmr',
                   'Compositions': 'compos',
                   'Placements(pie)': 'p',
                   'Placements(bar)': 'p_b',
                   'Top picks(nombre)': 'nombre de pick',
                   'Top pickrate': 'pickrate',
                   'Top propose(nombre)': 'nombre de fois proposé',
                   'Top gain MMR(champions)': 'mmr total gagné',
                   'Top gain MMR(comps)': 'mmr total gagné',
                   'Top perte MMR(champions)': 'mmr total perdu',
                   'Top perte MMR(comps)': 'mmr total perdue',
                   'Top Net MMR(champions)': 'gain mmr',
                   'Top Net MMR(comps)': 'net mmr',
                   'Top Winrate(champions)': 'winrate',
                   'Top Winrate(comps)': 'winrate',
                   'Top placement moyen(champions)': 'position moyenne',
                   'Top placement moyen(comps)': 'position moyenne',
                   'Top top 1 rate(champions)': '% top 1',
                   'Top top 1 rate(comps)': '% top 1',
                   'Top victoire(champions)': 'v_abs',
                   'Top victoire(comps)': 'Victoires',
                   'Top top 1(champions)': '1_abs',
                   'Top top 1(comps)': 'Top 1',
                   }

@app.callback(Output('content', 'children'),
              [Input('main', 'value')])
def render_content(tab):
    if tab == 'global':
        return dbc.Container(fluid=True, children=[
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([html.H2('Type de graphe'),
                                  dcc.Dropdown(id='t_1', options=[
                                               {'label': k, 'value': k} for k, v in graphs_generals.items()]),
                                  html.Div(
                            'nombre de perso'),
                            dcc.Input(id='n_max', type="number", value=5)]), width=6,
                        style={"height": "150px"},
                    ),
                    dbc.Col(
                        html.Div([html.H2('Type de graphe'),
                                  dcc.Dropdown(id='t_2', options=[
                                               {'label': k, 'value': k} for k, v in graphs_generals.items()]),
                                  html.Div(
                            'Nombre de pick mini'),
                            dcc.Input(id='n_min_2', type="number", value=3)]),
                        width=6,
                        style={"height": "150px"},
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id='g1'),
                        width=6,
                        style={"height": "850px"},
                    ),
                    dbc.Col(
                        html.Div(id='g2'),
                        width=6,
                        style={"height": "850px"},
                    ),
                ],
            ),
        ],
            style={"height": "1080px"},
        )

    elif tab == 'all':
        return html.Div(children=[
            dbc.Row([
                dbc.Col([html.H2('Colonnes a garder'), dcc.Dropdown(id='filtre_col',
                                                                    options=[{'label': v, 'value': v} for v in df_all_champ.columns], multi=True, value=[v for v in df_all_champ.columns]),
                         html.H2('sort by'), dcc.Dropdown(id='sort',
                                                          options=[{'label': v, 'value': v} for v in df_all_champ.columns]),
                         html.H2('Nombre de pick mini'),
                         dcc.Input(id='n_min', type="number", value=1)], width=3),
                dbc.Col(html.Div(id='content_table'), width=9)])])
    elif tab == 'solo':
        return html.Div([dcc.Dropdown(id='choix_perso',
                                      options=[{'label': k, 'value': k} for k in df_stats_champs['nom'].values]),
                         html.Div(id='graph_char')])
    elif tab == 'comps':
        return html.Div([dcc.Dropdown(id='choix_type',
                                      options=[{'label': k, 'value': k} for k in ['general']+list(df_types['nom'].values)]),
                         html.Div(id='graph_type')])

    elif tab == 'compar':
        return html.Div(children=[
            dbc.Row([
                dbc.Col([html.H2('Colonnes a garder'), dcc.Dropdown(id='filtre_col_c',
                                                                    options=[{'label': v, 'value': v} for v in df_all_champ.columns], multi=True, value=[v for v in df_all_champ.columns]),
                         html.H2('sort by'), dcc.Dropdown(id='sort_c',
                                                          options=[{'label': v, 'value': v} for v in df_all_champ.columns], value='position moyenne'),
                         html.H2('Champions'),
                         dcc.Dropdown(id='champ_sel', options=[
                                      {'label': k, 'value': k} for k in df_stats_champs['nom'].values], multi=True),
                         dcc.RadioItems(id='compar_type', options=[
                             {'label': 'Tableau', 'value': 'table'},
                             {'label': 'Graphes', 'value': 'graph'}
                         ], value='table')]),
                dbc.Col(html.Div(id='content_table_c'), width=9)])])


@app.callback(Output('g1', 'children'),
              [Input('t_1', 'value'),
               Input('n_max', 'value'),
               Input('n_min_2', 'value'),
               ])
def render_general_page1(t_1, n_max, n_min):
    return render_general_page(t_1, n_max, n_min)


@app.callback(Output('g2', 'children'),
              [Input('t_2', 'value'),
               Input('n_max', 'value'),
               Input('n_min_2', 'value')])
def render_general_page2(t_1, n_max, n_min):
    return render_general_page(t_1, n_max, n_min)


@app.callback(Output('graph_char', 'children'),
              [Input('choix_perso', 'value')])
def render_char_graph(char):
    if char == None:
        return html.Div('')
    if char not in df_stats_champs['nom'] or char not in df_top_champs['nom'] or char not in all_matches_champs.keys():
        return html.Div('Pas de parties !')
    all_types_mmr = np.array(
        list(comp_types_per_champ[char]['net_mmr'].keys()))
    all_types_pos = np.array(
        list(comp_types_per_champ[char]['position'].keys()))
    net_mmr_per_type = np.array(
        list(comp_types_per_champ[char]['net_mmr'].values()))
    mean_pos_per_type = np.array([round_(np.mean([int(a) for a in v]), 2)
                                  for k, v in comp_types_per_champ[char]['position'].items()])
    sorted_idx_mmr = np.argsort(-net_mmr_per_type)
    sorted_idx_pos = np.argsort(mean_pos_per_type)
    layout = html.Div([
        dbc.Row([
                dbc.Col(render_graph(x=list(range(len(all_matches_champs[char])+1)),
                                     y=[0]+list(all_matches_champs[char]
                                                ['gain mmr total'].values),
                                     titre=f"Gain moyen/total/perte totale : {df_all_champ.loc[char]['mmr moyen par partie']}/{df_all_champ.loc[char]['mmr total gagné']}/{df_all_champ.loc[char]['mmr total perdu']}",
                                     t='scatter')),
                dbc.Col(render_graph(x=[d.replace('%', '') for d in df_top_champs.columns[1:-1]],
                                     y=[round(x) for x in df_top_champs.loc[char].values[1:-1]
                                        * df_stats_champs.loc[char]['nombre de pick']/100],
                                     titre=f"winrate = {df_all_champ.loc[char]['winrate']}%, position moyenne =  {df_all_champ.loc[char]['position moyenne']}", t='bar_p'))
                ]
                ), dbc.Row([
                    dbc.Col(render_graph(x=list(range(len(cbt_winrate_champs[char])+1)),
                                         y=list(cbt_winrate_champs[char]),
                                         titre=f"Combat winrate",
                                         t='scatter')),

                    dbc.Col(render_graph(x=list(comp_types_per_champ[char]['types'].keys()),
                                         y=list(
                                             comp_types_per_champ[char]['types'].values()),
                                         titre='Type de compositions joués',
                                         t='pie_p'
                                         )),

                ]), dbc.Row([
                    dbc.Col(render_graph(x=np.take_along_axis(all_types_mmr, sorted_idx_mmr, axis=0),
                                         y=np.take_along_axis(
                                             net_mmr_per_type, sorted_idx_mmr, axis=0),
                                         titre='Net mmr par type',
                                         t='bar'

                                         )),
                    dbc.Col(render_graph(x=np.take_along_axis(all_types_pos, sorted_idx_pos, axis=0),
                                         y=np.take_along_axis(
                                             mean_pos_per_type, sorted_idx_pos, axis=0),
                                         titre='Mean position par type',
                                         t='bar'

                                         ))
                ]), dbc.Row([
                    dbc.Col([html.H2(f"pickrate = {df_all_champ.loc[char]['pickrate']}, % proposé : {df_all_champ.loc[char]['% proposé']} % "),
                             df2table_simple(all_matches_champs[char])])
                ])


    ])

    return layout


@app.callback(Output('content_table', 'children'),
              [Input('filtre_col', 'value'),
               Input('sort', 'value'),
               Input('n_min', 'value')])
def df2table(filter_columns, sort_by, n_min):
    n_min = 0 if not n_min else n_min
    dataframe = df_all_champ
    accepted_rows = list(dataframe['nom'].values)+['moyenne']
    accepted_cols = dataframe.columns if not filter_columns else filter_columns
    asc = ['position moyenne', 'mmr total perdu']
    ordered_columns = ['nom', 'position moyenne', 'winrate', 'mmr moyen par partie',  'gain mmr', 'mmr total gagné',
                       'mmr total perdu', 'pickrate', 'nombre de pick',
                       '% top 1', '% top 2', '% top 3', '% top 4', '% top 5', '% top 6',
                       '% top 7', '% top 8', 'nombre de fois proposé', '% de parties', '% proposé']
    dataframe = dataframe[ordered_columns]
    if sort_by != None:
        dataframe = dataframe.sort_values(
            sort_by, ascending=False if sort_by not in asc else True)
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col)
                     for col in dataframe.columns if col in accepted_cols])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns if col in accepted_cols
            ]) for i in range(len(dataframe)) if dataframe.iloc[i]['nom'] in accepted_rows and (dataframe.iloc[i]['nombre de pick'] >= n_min or dataframe.iloc[i]['nom'] == 'moyenne')
        ])
    ], className='table')

@app.callback(Output('graph_type', 'children'),
              [Input('choix_type', 'value')])
def render_type_page(choice):
    if choice is None:
        return 'Selectionnez un type dans la liste ci dessus'
    if choice == 'general':
        return df2table_simple(df_types)
    else:
        data = comp_types[choice]
        all_champs = np.array(list(data['champs_stats'].keys()))
        mmr_gain_per_champ = np.array([d['mmr']
                                       for d in data['champs_stats'].values()])

        mean_pos_per_champ = np.array([round_(np.mean([int(a) for a in d['pos']]), 2)
                                       for d in data['champs_stats'].values()])
        sort_idx_mmr = np.argsort(-mmr_gain_per_champ)
        sort_idx_pos = np.argsort(mean_pos_per_champ)
        layout = html.Div([
            dbc.Row([
                dbc.Col(render_graph(x=list(range(len(data['mmr_evo'])+1)),
                                     y=[0]+data['mmr_evo'],
                                     titre=f"Gain moyen/total/perte totale : {round_(data['mean mmr'],0)}/{data['gain mmr absolu']}/{data['perte mmr absolu']}",
                                     t='scatter')),

                dbc.Col(render_graph(x=[f'% top {i}' for i in range(1, 9)],
                                     y=[data[f'% top {i}']+0.00001
                                         for i in range(1, 9)],
                                     titre=f"winrate = {data['winrate']*100}%, position moyenne =  {round_(data['placement moyen'],2)}", t='bar_p'))
            ]
            ), dbc.Row([
                dbc.Col(render_graph(x=list(range(len(data['cbt_winrate'])+1)),
                                     y=data['cbt_winrate'],
                                     titre=f"Combat winrate",
                                     t='scatter')),
                dbc.Col(render_graph(x=list(data['champs_stats'].keys()),
                                     y=[d['nb_played']/data['nombre de pick']
                                         for d in data['champs_stats'].values()],
                                     titre='Repartition des persos',
                                     t='pie_p',
                                     ))]),
            dbc.Row([
                    dbc.Col(render_graph(
                        x=np.take_along_axis(all_champs, sort_idx_mmr, axis=0),
                        y=np.take_along_axis(
                            mmr_gain_per_champ, sort_idx_mmr, axis=0),
                        titre='Net MMR par perso',
                        t='bar'
                    )),
                    dbc.Col(render_graph(
                        x=np.take_along_axis(all_champs, sort_idx_pos, axis=0),
                        y=np.take_along_axis(
                            mean_pos_per_champ, sort_idx_pos, axis=0),
                        titre='Position moyenne par perso',
                        t='bar'))
                    ])



        ])
        return layout


@app.callback(Output('content_table_c', 'children'),
              [Input('filtre_col_c', 'value'),
               Input('sort_c', 'value'),
               Input('champ_sel', 'value'),
               Input('compar_type', 'value')])
def render_comparison(filtre_col, sort_by, champs, compar_type):
    if champs == None:
        return html.Div('Selectionnez un personnage')
    if compar_type == 'table':
        sort = ['nom', 'position moyenne', 'winrate', 'mmr moyen par partie',  'gain mmr', 'mmr total gagné',
                'mmr total perdu', 'pickrate', 'nombre de pick',
                '% top 1', '% top 2', '% top 3', '% top 4', '% top 5', '% top 6',
                '% top 7', '% top 8']
        dataframe = df_all_champ[sort]
        accepted_rows = champs
        accepted_cols = df_all_champ.columns if not filtre_col else filtre_col
        asc = ['position moyenne', 'mmr total perdu']
        if sort_by != None:
            dataframe = dataframe.sort_values(
                sort_by, ascending=False if sort_by not in asc else True)
        return html.Table([
            html.Thead(
                html.Tr([html.Th(col)
                         for col in dataframe.columns if col in accepted_cols])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns if col in accepted_cols
                ]) for i in range(len(dataframe)) if dataframe.iloc[i]['nom'] in accepted_rows
            ])
        ], className='table')
    elif compar_type == 'graph':
        gains_mmr = {champ: all_matches_champs[champ]
                     ['gain mmr total'].values for champ in champs}
        placement_moyen = {
            champ: df_all_champ.loc[champ]['position moyenne'] for champ in champs}
        placement_sorted = {k: v for k, v in sorted(
            placement_moyen.items(), key=lambda x: x[1], reverse=False)}
        placement_x = list(placement_sorted.keys())
        placement_y = list(placement_sorted.values())
        winrate = {
            champ: df_all_champ.loc[champ]['winrate'] for champ in champs}
        winrate_sorted = {k: v for k, v in sorted(
            winrate.items(), key=lambda x: x[1], reverse=True)}
        winrate_x = list(winrate_sorted.keys())
        winrate_y = list(winrate_sorted.values())
        return html.Div(children=[
            dbc.Row(children=[dbc.Col([html.H2('Position moyenne'), render_graph(placement_x, placement_y, t='bar_c', fig_title='Position moyenne')]),
                              dbc.Col(dcc.Graph(
                                  id='compar_mmr',
                                  figure=go.Figure(data=[go.Scatter(name=champ, y=[0]+list(gains_mmr[champ]), x=[*range(len(gains_mmr[champ])+1)]) for champ in champs],
                                                   layout=go.Layout(title=go.layout.Title(
                                                       text=' <b>Gain de MMR selon les matchs </b>'))

                                                   )))


                              ]),
            dbc.Row(children=[
                dbc.Col([html.H2('Winrate'), render_graph(winrate_x, winrate_y,
                                                          t='bar_c', fig_title='Winrate')]),
                dbc.Col(dcc.Graph(
                    id='compar_mmr',
                    figure=go.Figure(data=[go.Scatter(name=champ, y=[0]+cbt_winrate_champs[champ], x=[*range(len(cbt_winrate_champs[champ])+1)]) for champ in champs],
                                     layout=go.Layout(title=go.layout.Title(
                                         text=' <b>Combat winrate</b>'))

                                     )))
            ])])


def render_general_page(t_1, n_max, n_min):
    if t_1 == None:
        pass
    elif t_1 == 'MMR':
        return dcc.Graph(
            id='example-graph',
            figure=go.Figure(data=[go.Scatter(y=mmr, x=list(range(len(mmr))))], layout={'margin': {'t': 0, 'l': 0}}))
    elif t_1 == 'MMR avec moyenne':
        return dcc.Graph(
            id='example-graph',
            figure=go.Figure(data=[go.Scatter(name='MMR', y=mmr, x=list(range(len(mmr)))),
                                   go.Scatter(name='moyenne sur 10 parties', y=rolling_mean(
                                       mmr, 10), x=list(range(len(mmr)))),
                                   go.Scatter(name='moyenne sur 25 parties', y=rolling_mean(mmr, 25), x=list(range(len(mmr))))],
                             layout={'margin': {'t': 0, 'l': 0}}))
    elif t_1 == 'Compositions':
        return render_graph(x=df_types['nom'].values,
                            y=df_types['Nb de fois joué'].values,
                            titre='Compositions jouées',
                            t='pie')
    elif t_1 == 'Top top 1(champions)':
        results = {}
        for champ in df_stats_champs['nom']:
            if not np.isnan(df_all_champ.loc[champ]['nombre de pick']) and not np.isnan(df_all_champ.loc[champ]['% top 1']):
                results[champ] = round(
                    df_all_champ.loc[champ]['nombre de pick']*df_all_champ.loc[champ]['% top 1']/100)
        sort_res = {k: v for k, v in sorted(
            results.items(), key=lambda x: x[1], reverse=True)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar'))
    elif t_1 == 'Top victoire(champions)':
        results = {}
        for champ in df_stats_champs['nom']:
            if not np.isnan(df_all_champ.loc[champ]['nombre de pick']) and not np.isnan(df_all_champ.loc[champ]['winrate']):
                results[champ] = round(
                    df_all_champ.loc[champ]['nombre de pick']*df_all_champ.loc[champ]['winrate']/100)

        sort_res = {k: v for k, v in sorted(
            results.items(), key=lambda x: x[1], reverse=True)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar'))
    elif t_1 == 'Placements(pie)':
        nb_parties = sum(
            [x for x in df_stats_champs['nombre de pick'].values if not np.isnan(x)])
        x = [d.replace('%', '') for d in df_top_champs.columns[1:-1]]
        y = [round(x)
             for x in df_top_champs.loc['global'].values[1:-1]*nb_parties/100]
        return html.Div(
            render_graph(
                x, y, titre='',
                fig_title=f"Winrate global : {df_top_champs.loc['global']['winrate']}<br>, Placement moyen : {mean_position}"))
    elif t_1 == 'Placements(bar)':
        nb_parties = sum(
            [x for x in df_stats_champs['nombre de pick'].values if not np.isnan(x)])
        x = [d.replace('%', '') for d in df_top_champs.columns[1:-1]]
        y = [round(x)
             for x in df_top_champs.loc['global'].values[1:-1]*nb_parties/100]
        return html.Div(
            render_graph(
                x, y, titre='', t='bar_g',
                fig_title=f"Winrate global : {df_top_champs.loc['global']['winrate']}, Placement moyen : {mean_position}"))

    else:
        key = graphs_generals[t_1]
        results = {}
        if key not in ['position moyenne', 'gain mmr', 'pickrate', '% top 1', 'winrate'] or n_min == None:
            n_min = 0
        df = df_all_champ if not'comps' in t_1 else df_types
        name_df = df_stats_champs if not'comps' in t_1 else df_types
        n_max = n_max if not'comps' in t_1 else 10
        for champ in name_df['nom']:
            if not np.isnan(df.loc[champ][key]) and df.loc[champ]['nombre de pick'] >= n_min:
                results[champ] = df.loc[champ][key]
        sort_res = {k: v for k, v in sorted(results.items(
        ), key=lambda x: x[1], reverse=True if key not in ['position moyenne'] else False)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar'))


def render_graph(x, y, titre='', t='pie', fig_title='', **kwargs):
    # General pie
    if t == 'pie':
        colors = ['#2ED9FF', '#17BECF', '#00B5F7', '#2E91E5',
                  '#FBE426', '#FEAF16', '#FD3216', '#DC3912']
        return html.Div(style={'height': '70vh'}, children=[
                        html.H2(titre),
                        dcc.Graph(figure=go.Figure(data=[go.Pie(labels=x, values=y, textinfo='label+percent+value', marker=go.Marker(colors=colors))],
                                                   layout=go.Layout(title=go.layout.Title(text=fig_title),
                                                                    margin={'t': 50, 'l': 0})
                                                   ),
                                  style={'height': 'inherit'})

                        ])
    # Pie per char /type
    elif t == 'pie_p':
        return html.Div(style={'height': '50vh'}, children=[
                        html.H2(titre),
                        dcc.Graph(figure=go.Figure(data=[go.Pie(labels=x, values=y, textinfo='label+percent')],
                                                   layout=go.Layout(margin={'t': 0, 'l': 0},
                                                                    title=go.layout.Title(text=fig_title))),
                                  style={'height': 'inherit'})

                        ])
    # Bar graphs with images
    elif t == 'bar':
        char = x[0]
        if char in imgs:
            images = get_img_dict([imgs[char] for char in x])
        else:
            images = get_img_dict([imgs_types[t] for t in x])
        return html.Div(style={'height': '90vh'}, children=[
            html.H2(titre),
            dcc.Graph(style={'height': 'inherit'}, figure=go.Figure(data=[go.Bar(x=y, text=y, textposition='auto', orientation='h')],
                                                                    layout=go.Layout(margin={'t': 0, 'l': 0},
                                                                                     title=go.layout.Title(text=fig_title), yaxis=dict(autorange="reversed", showticklabels=False), images=images)))])
    # Bar graph for char page
    elif t == 'bar_p':
        return html.Div(style={'height': '50vh'}, children=[
                        html.H2(titre),
                        dcc.Graph(figure=go.Figure(data=[go.Bar(x=x, y=y, text=[round(i*100/sum(y)) for i in y], textposition='outside')],
                                                   layout=go.Layout(margin={'t': 0, 'l': 0},
                                                                    title=go.layout.Title(text=fig_title))),
                                  style={'height': 'inherit'})])
    # Bar graph for overall placement
    elif t == 'bar_g':
        return html.Div(style={'height': '50vh'}, children=[
                        html.H2(titre),
                        dcc.Graph(figure=go.Figure(data=[go.Bar(x=x, y=y, text=[round(i*100/sum(y)) for i in y], textposition='outside')],
                                                   layout=go.Layout(margin={'t': 0, 'l': 0},
                                                                    title=go.layout.Title(text=fig_title, y=0))),
                                  style={'height': 'inherit'})])
    # Bar graph for comparison
    elif t == 'bar_c':
        images = get_img_dict([imgs[char] for char in x])
        return html.Div(style={'height': '40vh'}, children=[
            html.H2(titre),
            dcc.Graph(style={'height': 'inherit'}, figure=go.Figure(data=[go.Bar(x=y, text=y, textposition='auto', orientation='h')],
                                                                    layout=go.Layout(margin={'t': 0, 'l': 0},
                                                                                     title=go.layout.Title(text=fig_title), yaxis=dict(autorange="reversed", showticklabels=False), images=images)))])
    # 2 bars ( not used)
    elif t == '2bar':
        return html.Div([
            html.H2(titre),
            dcc.Graph(figure=go.Figure(data=[go.Bar(name='nouveau', x=x, y=y[0], text=y[0], textposition='auto'),
                                             go.Bar(name='total', x=x, y=y[1], text=y[1], textposition='auto')],
                                       layout=go.Layout(margin={'t': 0, 'l': 0},
                                                        title=go.layout.Title(text=fig_title))))])
    # Line plots
    elif t == 'scatter':
        return html.Div([
            html.H2(titre),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=x, y=y, text=y, textposition='top center')],
                                       layout=go.Layout(margin={'t': 0, 'l': 0},
                                                        title=go.layout.Title(text=fig_title))))])


def df2table_simple(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in reversed(range(len(dataframe)))
        ])
    ], className='table')


def get_img_dict(images):
    return [{'source': img,
             'layer': 'below',
             'x': 1,
             'y': i-0.3,
             'sizex': 0.5,
             'sizey': 0.8,
             'xref': 'paper',
             'yref': 'y',
             } for i, img in enumerate(images)]


if __name__ == '__main__':
    app.run_server(debug=False)
