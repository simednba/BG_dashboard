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
from compute_stats import get_all_stats, rolling_mean
import dash_table
import numpy as np
from PIL import Image
from config import IMG_PATH


imgs = {}
for filename in os.listdir(IMG_PATH):
    if filename.lower().endswith('.png'):
        path = os.path.join(IMG_PATH, filename)
        imgs[filename[:-4]] = Image.open(path)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

df_stats_champs, df_top_champs, df_all_champ, all_matches_champs, mmr, mean_position, cbt_winrate_champs, comp_types_per_champ, comp_types, battle_luck = get_all_stats()
df_all_champ.loc['global', df_stats_champs.describe(
).columns] = df_stats_champs.describe().loc['mean'].round(2)
df_all_champ.loc['global', 'nom'] = 'moyenne'


app.layout = html.Div(children=[dcc.Tabs(id='main', value='main_v', children=[
    dcc.Tab(label='Global', value='global'),
    dcc.Tab(label='Toute les Données', value='all'),
    dcc.Tab(label='Par Perso', value='solo'),
    dcc.Tab(label='Comparaison', value='compar'),
]),
    html.Div(id='content'),


])

graphs_generals = {'MMR': 'mmr',
                   'MMR avec moyenne': 'mean_mmr',
                   'Placements(pie)': 'p',
                   'Placements(bar)': 'p_b',
                   'Top picks(nombre)': 'nombre de pick',
                   'Top pickrate': 'pickrate',
                   'Top propose(nombre)': 'nombre de fois proposé',
                   'Top gain MMR(absolu)': 'mmr total gagné',
                   'Top perte MMR': 'mmr total perdu',
                   'Top gain MMR(relatif)': 'gain mmr',
                   'Top Winrate': 'winrate',
                   'Top placement moyen': 'position moyenne',
                   'Top top 1 rate': '% top 1',
                   'Top victoire(absolu)': 'v_abs',
                   'Top top 1(absolu)': '1_abs',
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

    elif t_1 == 'Top top 1(absolu)':
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
    elif t_1 == 'Top victoire(absolu)':
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
        for champ in df_stats_champs['nom']:
            if not np.isnan(df_all_champ.loc[champ][key]) and df_all_champ.loc[champ]['nombre de pick'] >= n_min:
                results[champ] = df_all_champ.loc[champ][key]
        sort_res = {k: v for k, v in sorted(results.items(
        ), key=lambda x: x[1], reverse=True if key not in ['position moyenne'] else False)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar'))


@app.callback(Output('g2', 'children'),
              [Input('t_2', 'value'),
               Input('n_max', 'value'),
               Input('n_min_2', 'value')])
def render_general_page1(t_1, n_max, n_min):
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

    elif t_1 == 'Top top 1(absolu)':
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
    elif t_1 == 'Top victoire(absolu)':
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
        for champ in df_stats_champs['nom']:
            if not np.isnan(df_all_champ.loc[champ][key]) and df_all_champ.loc[champ]['nombre de pick'] >= n_min:
                results[champ] = df_all_champ.loc[champ][key]
        sort_res = {k: v for k, v in sorted(results.items(
        ), key=lambda x: x[1], reverse=True if key not in ['position moyenne'] else False)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar'))
@app.callback(Output('graph_char', 'children'),
              [Input('choix_perso', 'value')])
def render_char_graph(char):
    if char == None:
        return html.Div('')
    if char not in df_stats_champs['nom'] or char not in df_top_champs['nom'] or char not in all_matches_champs.keys():
        return html.Div('Pas de parties !')
    layout = html.Div([
        dbc.Row([
                dbc.Col(render_graph(x=list(range(len(all_matches_champs[char])+1)),
                                     y=[0]+list(all_matches_champs[char]
                                                ['gain mmr total'].values),
                                     titre='Gain MMR total selon les matchs',
                                     t='scatter')),
                dbc.Col(render_graph(x=[d.replace('%', '') for d in df_top_champs.columns[1:-1]],
                                     y=[round(x) for x in df_top_champs.loc[char].values[1:-1]
                                        * df_stats_champs.loc[char]['nombre de pick']/100],
                                     titre=f"Placements, winrate = {df_all_champ.loc[char]['winrate']}%, position moyenne =  {df_all_champ.loc[char]['position moyenne']}", t='bar_p'))
                ]
                ), dbc.Row([
                    dbc.Col(html.Div(dcc.Markdown(f'''
                                     # Stats générales
                                     - **Nombre de Parties** : {df_all_champ.loc[char]['nombre de pick']}
                                     - **Winrate** : {df_all_champ.loc[char]['winrate']}
                                     - **Position moyenne** : {df_all_champ.loc[char]['position moyenne']}
                                     - **Pickrate** : {df_all_champ.loc[char]['pickrate']}
                                     - **Gain moyen de MMR** : {df_all_champ.loc[char]['mmr moyen par partie']}
                                     - **Gain relatif de MMR** : {df_all_champ.loc[char]['gain mmr']}
                                     - **Gain total de MMR** : {df_all_champ.loc[char]['mmr total gagné']}
                                     - **Perte total de MMR** : {df_all_champ.loc[char]['mmr total perdu']}
                                     - **Proposé dans % de parties** : {df_all_champ.loc[char]['% proposé']}
                                     ''', className='md')), style={'textAlign': 'center', 'fontSize': '25px'}),
                    dbc.Col([html.H2('Dernieres parties'),
                             df2table_simple(all_matches_champs[char])]),

                ])])

    return layout


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
    # Pie per char ( Not used)
    elif t == 'pie_p':
        return html.Div(style={'height': '50vh'}, children=[
                        html.H2(titre),
                        dcc.Graph(figure=go.Figure(data=[go.Pie(labels=x, values=y, textinfo='label+percent+value')],
                                                   layout=go.Layout(margin={'t': 0, 'l': 0},
                                                                    title=go.layout.Title(text=fig_title))),
                                  style={'height': 'inherit'})

                        ])
    # Bar graphs with images
    elif t == 'bar':
        images = get_img_dict([imgs[char] for char in x])
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
                                                          t='bar_c', fig_title='Winrate')])
            ])])


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
