# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:24:02 2020

@author: simed
"""

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.graph_objs as go
from bg_logs_reader import get_all_stats
import dash_table
import numpy as np

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

df_stats, df_top,df_all,all_matches, mmr = get_all_stats()

app.layout = html.Div(children=[dcc.Tabs(id='main',value='main_v',children = [
    dcc.Tab(label='Global', value = 'global'),
    dcc.Tab(label = 'all_table', value='all'),
    dcc.Tab(label = 'par_perso', value = 'solo'),
    ])
    ,
    html.Div(id = 'content'),


])

graphs_generals = {'MMR' : 'mmr',
                   'Placements' : 'p',
                   'Top picks(nombre)' : 'nombre de pick',
                   'Top pickrate' : 'pickrate',
                   'Top propose(nombre)' : 'nombre de fois proposé',
                   'Top gain MMR(absolu)' : 'mmr total gagné',
                   'Top perte MMR' : 'mmr total perdu',
                   'Top gain MMR(relatif)' : 'gain mmr',
                   'Top Winrate' : 'winrate',
                   'Top placement moyen' : 'position moyenne',
                   'Top top 1 rate' : '% top 1',
                   'Top victoire(absolu)' :'v_abs',
                   'Top top 1(absolu)' : '1_abs',
                   }

@app.callback(Output('content', 'children'),
              [Input('main', 'value')])
def render_content(tab):
    if tab =='global':
       return dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div([html.H2('Type de graphe'),
                              dcc.Dropdown(id = 't_1', options = [{'label' : k, 'value' : k} for k,v in graphs_generals.items()]),
                              dcc.Input(id='n_max', type="number",value = 5)])
                    ,width=6,
                    style={"height": "100px"},
                ),
                dbc.Col(
                    html.Div([html.H2('Type de graphe'),
                              dcc.Dropdown(id = 't_2', options = [{'label' : k, 'value' : k} for k,v in graphs_generals.items()]),
                              dcc.Input(id='n_min_2', type="number",value = 3)])
                    ,
                    width=6,
                    style={"height": "100px"},
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id='g1'),
                    width=6,
                    style={"height": "900px"},
                ),
                dbc.Col(
                    html.Div(id='g2'),
                    width=6,
                    style={"height": "900px"},
                ),
            ]
            ,
        ),
    ],
    style={"height": "1080px"},
)

    elif tab =='all':
        return html.Div(children=[
                    dbc.Row([
                            dbc.Col([html.H2('Colonnes a garder'),dcc.Dropdown(id = 'filtre_col',
                                      options = [{'label' : v, 'value':v} for v in df_all.columns], multi = True, value=[v for v in df_all.columns]),
                                     html.H2('sort by'),dcc.Dropdown(id = 'sort',
                                      options = [{'label' : v, 'value':v} for v in df_all.columns]),
                                     html.H2('Nombre de pick mini'),
                                     dcc.Input(id='n_min', type="number",value = 1)], width = 3),
                            dbc.Col(html.Div(id='content_table'), width = 9)])])
    elif tab =='solo':
        return html.Div([dcc.Dropdown(id = 'choix_perso',
    options=[{'label' : k, 'value' : k} for k in df_stats['nom'].values]),
            html.Div(id = 'graph_char')])

@app.callback(Output('g1', 'children'),
              [Input('t_1', 'value'),
               Input('n_max','value'),
               Input('n_min_2','value'),
               ])
def render_general_page1(t_1, n_max, n_min):
    if t_1 =='MMR':
        return dcc.Graph(
                    id='example-graph',
                    figure= go.Figure(data=[go.Scatter(y=mmr, x=list(range(len(mmr))))], layout={'margin' : {'t' : 10}}))
    elif t_1 =='Top top 1(absolu)':
        results = {}
        for champ in df_stats['nom']:
            if not np.isnan(df_all.loc[champ]['nombre de pick'])  and not np.isnan(df_all.loc[champ]['% top 1']):
                results[champ] = round(df_all.loc[champ]['nombre de pick']*df_all.loc[champ]['% top 1']/100)
        sort_res = {k : v for k,v in sorted(results.items(), key = lambda x:x[1], reverse=True)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar', titre='Nombres de Top 1'))
    elif t_1 =='Top victoire(absolu)':
        results = {}
        for champ in df_stats['nom']:
            if not np.isnan(df_all.loc[champ]['nombre de pick'])  and not np.isnan(df_all.loc[champ]['winrate']):
                results[champ] = round(df_all.loc[champ]['nombre de pick']*df_all.loc[champ]['winrate']/100)
               
        sort_res = {k : v for k,v in sorted(results.items(), key = lambda x:x[1], reverse=True)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar', titre='Nombres de Victoires'))
    elif t_1 =='Placements':
        nb_parties = sum([x for x in df_all['nombre de pick'].values if not np.isnan(x)])
        x=df_top.columns[1:-1]
        y=[round(x) for x in df_top.loc['global'].values[1:-1]*nb_parties/100]
        return html.Div(render_graph(x,y,titre = 'Placements', fig_title = f'Winrate global : {df_top.loc["global"]["winrate"]}<br>, Placement moyen :{round(sum([x/100*(i+1) for i,x in enumerate(df_top.loc["global"].values[1:-1])]),2)} '))
    else:
        key = graphs_generals[t_1]
        results = {}
        print(key)
        if key not in ['position moyenne','gain mmr','pickrate', '% top 1','winrate'] or n_min == None:
            n_min = 0
        for champ in df_stats['nom']:
            if not np.isnan(df_all.loc[champ][key]) and df_all.loc[champ]['nombre de pick']>=n_min:
                results[champ] = df_all.loc[champ][key]      
        sort_res = {k : v for k,v in sorted(results.items(), key = lambda x:x[1], reverse=True if key not in ['position moyenne'] else False)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar', titre=t_1))

@app.callback(Output('g2', 'children'),
              [Input('t_2', 'value'),
               Input('n_max','value'),
               Input('n_min_2','value')])
def render_general_page1(t_1, n_max, n_min):
    if t_1 =='MMR':
        return dcc.Graph(
                    id='example-graph',
                    figure= go.Figure(data=[go.Scatter(y=mmr, x=list(range(len(mmr))))], layout={'margin' : {'t' : 10}}))
    elif t_1 =='Top top 1(absolu)':
        results = {}
        for champ in df_stats['nom']:
            if not np.isnan(df_all.loc[champ]['nombre de pick'])  and not np.isnan(df_all.loc[champ]['% top 1']):
                results[champ] = round(df_all.loc[champ]['nombre de pick']*df_all.loc[champ]['% top 1']/100)
        sort_res = {k : v for k,v in sorted(results.items(), key = lambda x:x[1], reverse=True)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar', titre='Nombres de Top 1'))
    elif t_1 =='Top victoire(absolu)':
        results = {}
        for champ in df_stats['nom']:
            if not np.isnan(df_all.loc[champ]['nombre de pick'])  and not np.isnan(df_all.loc[champ]['winrate']):
                results[champ] = round(df_all.loc[champ]['nombre de pick']*df_all.loc[champ]['winrate']/100)
               
        sort_res = {k : v for k,v in sorted(results.items(), key = lambda x:x[1], reverse=True)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar', titre='Nombres de Victoires'))
    elif t_1 =='Placements':
        nb_parties = sum([x for x in df_all['nombre de pick'].values if not np.isnan(x)])
        x=df_top.columns[1:-1]
        y=[round(x) for x in df_top.loc['global'].values[1:-1]*nb_parties/100]
        return html.Div(render_graph(x,y,titre = 'Placements', fig_title = f'Winrate global : {df_top.loc["global"]["winrate"]}<br>, Placement moyen :{round(sum([x/100*(i+1) for i,x in enumerate(df_top.loc["global"].values[1:-1])]),2)} '))
    else:
        key = graphs_generals[t_1]
        results = {}
        if key not in ['position moyenne','gain mmr','pickrate', '% top 1','winrate'] or n_min ==None:
            n_min = 0
        for champ in df_stats['nom']:
            if not np.isnan(df_all.loc[champ][key]) and df_all.loc[champ]['nombre de pick']>=n_min:
                results[champ] = df_all.loc[champ][key]      
        sort_res = {k : v for k,v in sorted(results.items(), key = lambda x:x[1], reverse=True if key not in ['position moyenne'] else False)}
        x = list(sort_res.keys())[:n_max]
        y = list(sort_res.values())[:n_max]
        return html.Div(render_graph(x, y, t='bar', titre=t_1))   
@app.callback(Output('graph_char', 'children'),
              [Input('choix_perso', 'value')])
def render_char_graph(char):
    if char == None:
        return html.Div('')
    if char not in df_stats['nom'] or char not in df_top['nom'] or char not in all_matches.keys():
        return html.Div('Pas de parties !')
    layout= html.Div([
            dbc.Row([
                dbc.Col(render_graph(x=['mmr gagné', 'mmr perdu'],
                                     y=[df_stats.loc[char]['mmr total gagné'],df_stats.loc[char]['mmr total perdu']],
                                     titre='MMR',
                                     t = 'bar')),                                                       
                dbc.Col(render_graph(x=df_top.columns[1:-1],
                                     y=[round(x) for x in df_top.loc[char].values[1:-1]*df_stats.loc[char]['nombre de pick']/100],
                                     titre='Placements'))
                
          ]         
        ), dbc.Row([
                dbc.Col(dcc.Markdown(f'''
                                     ### Stats générales
                                     - **Nombre de Parties** : {df_all.loc[char]['nombre de pick']}
                                     - **Winrate** : {df_all.loc[char]['winrate']}
                                     - **Position moyenne** : {df_all.loc[char]['position moyenne']}
                                     - **Proposé dans % de parties** : {df_all.loc[char]['% proposé']}
                                     - **Pickrate** : {df_all.loc[char]['pickrate']}
                                     - **Gain moyen de MMR** : {df_all.loc[char]['mmr moyen par partie']}
                                     - **Gain total de MMR** : {df_all.loc[char]['gain mmr']}
                                     ''', className = 'md')),                                                       
                dbc.Col([html.H2('Dernieres parties'),df2table_simple(all_matches[char])]),
                
                    ])])
            
        
    return layout

    
def render_graph(x, y , titre, t='pie', fig_title = '' ):
    if t =='pie':
        return html.Div(style = {'height' : '50vh'},children=[
                        html.H2(titre),
                        dcc.Graph(figure = go.Figure(data=[go.Pie(labels= x , values= y ,textinfo='label+percent+value')],
                                                     layout = go.Layout(title = go.layout.Title(text=fig_title))), 
                                  style = {'height' : 'inherit'})
                        
                            ])
    elif t =='bar':
        return html.Div(style = {'height' : '50vh'},children=[
            html.H2(titre),
            dcc.Graph(style = {'height' : 'inherit'},figure= go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition = 'auto')],
                                        layout = go.Layout(title = go.layout.Title(text=fig_title))))])
    elif t =='2bar':
        return html.Div([
            html.H2(titre),
            dcc.Graph(figure= go.Figure(data=[go.Bar(name='nouveau',x=x, y=y[0], text=y[0], textposition = 'auto'),
                                              go.Bar(name='total',x=x, y=y[1], text=y[1], textposition = 'auto')],
                                        layout = go.Layout(title = go.layout.Title(text=fig_title))))])
        
@app.callback(Output('content_table', 'children'),
              [Input('filtre_col', 'value'),
               Input('sort', 'value'),
               Input('n_min','value')])  
def df2table(filter_columns, sort_by, n_min):
    n_min = 0 if not n_min else n_min
    dataframe = df_all
    accepted_rows = list(dataframe['nom'].values)
    accepted_cols = dataframe.columns if not filter_columns else filter_columns
    asc = ['position moyenne','mmr total perdu']
    if sort_by !=None:
        dataframe = dataframe.sort_values(sort_by, ascending = False if sort_by not in asc else True)
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns if col in accepted_cols])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns if col in accepted_cols
            ]) for i in range(len(dataframe)) if dataframe.iloc[i]['nom'] in accepted_rows and dataframe.iloc[i]['nombre de pick'] >=n_min
        ])
    ],className = 'table')

def df2table_simple(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns 
            ]) for i in reversed(range(len(dataframe))) 
        ])
    ], className='table')

if __name__ == '__main__':
    app.run_server(debug=False)