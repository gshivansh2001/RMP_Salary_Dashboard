# Importing the required python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tqdm import tqdm
import warnings
from datetime import datetime
import ast
from tqdm.auto import tqdm
tqdm.pandas()
import streamlit as st
import plotly.express as px
import dash
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask import Flask
from dash_extensions.enrich import DashProxy
from dash import html
from base64 import b64encode
import io

# Sample data preparation - replace this with your actual data loading
df_ipeds = pd.read_csv('F:\\Shivansh Gupta\\OneDrive\\OneDrive - Indian School of Business\\MeToo Project\\Jupyter Code Files\\RMP Data Files\\RMP Data Files\\IPEDS_Quality_Metrics1.csv', low_memory = False)
df_ipeds['Owner'] = df_ipeds['Owner'].replace({'PrivateP': 'Private', 'PrivateNP': 'Private'})
df_ipeds_agg = df_ipeds.groupby(['Year', 'Owner', 'State_Category'])[['AllSalary_Total', 'AllSalary_Male', 'AllSalary_Female', 'ProfSalary_Total', 'ProfSalary_Male', 'ProfSalary_Female', 'Asc_ProfSalary_Total', 'Asc_ProfSalary_Male', 'Asc_ProfSalary_Female', 'Ast_ProfSalary_Total', 'Ast_ProfSalary_Female', 'Ast_ProfSalary_Male']].mean().reset_index()
# Melt DataFrame to have a 'Gender' column instead of separate salary columns
df_ipeds_agg = df_ipeds_agg.melt(id_vars=['Year', 'Owner', 'State_Category'], 
                         value_vars=['AllSalary_Male', 'AllSalary_Female', 'ProfSalary_Male', 'ProfSalary_Female', 'Asc_ProfSalary_Male', 'Asc_ProfSalary_Female', 'Ast_ProfSalary_Female', 'Ast_ProfSalary_Male'], 
                         var_name='Gender', value_name='Salary')
df_ipeds_agg['Gender'] = df_ipeds_agg['Gender'].replace({'AllSalary_Male': 'All Staff_Male', 'AllSalary_Female': 'All Staff_Female', 'ProfSalary_Male': 'Professor_Male', 'ProfSalary_Female': 'Professor_Female', 'Asc_ProfSalary_Male': 'Associate Professor_Male', 'Asc_ProfSalary_Female': 'Associate Professor_Female', 'Ast_ProfSalary_Female': 'Assistant Professor_Female', 'Ast_ProfSalary_Male':  'Assistant Professor_Male'})

df_ipeds_agg['Gender'] = df_ipeds_agg['Gender'].apply(lambda x: x.split('_'))
df_ipeds_agg['Rank'] = df_ipeds_agg['Gender'].apply(lambda x: x[0])
df_ipeds_agg['Gender'] = df_ipeds_agg['Gender'].apply(lambda x: x[1])
data = df_ipeds_agg[['Year', 'Owner', 'State_Category', 'Gender', 'Rank', 'Salary']]

# Create Flask server
server = Flask(__name__)


# Function to save HTML
def save_dash_html(filename="dashboard.html"):
    with server.test_client() as client:
        response = client.get("/")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.data.decode())

# Create the Dash app
app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server = server, prevent_initial_callbacks=True)

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("University Salary Dashboard (2012-2023)", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Year Range:"),
            dcc.RangeSlider(
                id='year-slider',
                min=data['Year'].min(),
                max=data['Year'].max(),
                step=1,
                marks={str(year): str(year) for year in range(data['Year'].min(), data['Year'].max()+1)},
                value=[data['Year'].min(), data['Year'].max()]
            ),
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select University Type:"),
            dcc.Checklist(
                id='owner-checklist',
                options=[{'label': owner, 'value': owner} for owner in data['Owner'].unique()],
                value=data['Owner'].unique().tolist(),  # Select all by default
                inline=True
            ),
        ], width=6),
        
        dbc.Col([
            html.Label("Select State Category:"),
            dcc.Checklist(
                id='state-checklist',
                options=[{'label': state, 'value': state} for state in data['State_Category'].unique()],
                value=data['State_Category'].unique().tolist(),  # Select all by default
                inline=True
            ),
        ], width=6),
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Gender:"),
            dcc.Checklist(
                id='gender-checklist',
                options=[{'label': gender, 'value': gender} for gender in data['Gender'].unique()],
                value=data['Gender'].unique().tolist(),  # Select all by default
                inline=True
            ),
        ], width=6),
        
        dbc.Col([
            html.Label("Select Rank:"),
            dcc.Checklist(
                id='rank-checklist',
                options=[{'label': rank, 'value': rank} for rank in data['Rank'].unique()],
                value=data['Rank'].unique().tolist(),  # Select all by default
                inline=True
            ),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Compare by:"),
            dcc.Checklist(
                id='compare-checklist',
                options=[
                    {'label': 'University Type', 'value': 'Owner'},
                    {'label': 'State Category', 'value': 'State_Category'},
                    {'label': 'Gender', 'value': 'Gender'},
                    {'label': 'Rank', 'value': 'Rank'}
                ],
                value=[],  # Default: no comparison
                inline=True
            ),
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='salary-graph')
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='stats-output', className="mt-4")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.A(
                html.Button("Download as HTML", className="btn btn-primary"),
                id="download",
                href="data:text/html;base64,",
                download="salary_dashboard.html"
            )
        ], width=12, className="text-center mt-3")
    ])
], fluid=True)

# Define callback to update the graph
@app.callback(
    [Output('salary-graph', 'figure'),
     Output('stats-output', 'children')],
    [Input('year-slider', 'value'),
     Input('owner-checklist', 'value'),
     Input('state-checklist', 'value'),
     Input('gender-checklist', 'value'),
     Input('rank-checklist', 'value'),
     Input('compare-checklist', 'value')]
)
def update_graph(year_range, owners, states, genders, ranks, compare_by_list):
    # Filter the data based on the selected options
    filtered_data = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]
    
    # Apply filters only if values are selected
    if owners:
        filtered_data = filtered_data[filtered_data['Owner'].isin(owners)]
    
    if states:
        filtered_data = filtered_data[filtered_data['State_Category'].isin(states)]
    
    if genders:
        filtered_data = filtered_data[filtered_data['Gender'].isin(genders)]
    
    if ranks:
        filtered_data = filtered_data[filtered_data['Rank'].isin(ranks)]
    
    # Create the figure
    if not compare_by_list:  # If no comparison variables selected
        # Group by year and calculate mean salary
        year_salary = filtered_data.groupby('Year')['Salary'].mean().reset_index()
        fig = px.line(year_salary, x='Year', y='Salary', 
                      title='Average Salary Trend Over Time',
                      labels={'Salary': 'Average Salary ($)', 'Year': 'Year'},
                      markers=True)
    else:
        # Combine multiple comparison variables
        if len(compare_by_list) == 1:
            # If only one variable selected, use it directly
            compare_by = compare_by_list[0]
            grouped_data = filtered_data.groupby(['Year', compare_by])['Salary'].mean().reset_index()
            
            # Use Plotly's built-in coloring system - no need to specify custom colors
            fig = px.line(grouped_data, x='Year', y='Salary', color=compare_by,
                          title=f'Average Salary Trend by {compare_by}',
                          labels={'Salary': 'Average Salary ($)', 'Year': 'Year'},
                          markers=True,
                          color_discrete_sequence=px.colors.qualitative.Plotly)  # Using Plotly's default color sequence
        else:
            # For multiple variables, create a combined category
            # Create a new column that combines the selected variables
            filtered_data['Combined'] = ''
            for var in compare_by_list:
                filtered_data['Combined'] += filtered_data[var].astype(str) + ' | '
            filtered_data['Combined'] = filtered_data['Combined'].str.rstrip(' | ')
            
            grouped_data = filtered_data.groupby(['Year', 'Combined'])['Salary'].mean().reset_index()
            
            # Use Plotly's built-in color sequences which are designed to be distinct
            fig = px.line(grouped_data, x='Year', y='Salary', color='Combined',
                          title=f'Average Salary Trend by {", ".join(compare_by_list)}',
                          labels={'Salary': 'Average Salary ($)', 'Year': 'Year', 'Combined': 'Categories'},
                          markers=True,
                          color_discrete_sequence=px.colors.qualitative.Bold)  # Using Bold color sequence for more distinction
        
        # Make lines thicker and markers larger for better visibility
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        
        # Enhance the legend
        fig.update_layout(
            legend=dict(
                title=dict(text='Categories' if len(compare_by_list) > 1 else compare_by_list[0]),
                orientation="h" if len(fig.data) < 8 else "v",  # Horizontal if few items, vertical otherwise
                yanchor="bottom",
                y=-0.3 if len(fig.data) < 8 else 1,
                xanchor="center" if len(fig.data) < 8 else "left",
                x=0.5 if len(fig.data) < 8 else 1.05,
                font=dict(size=12)
            )
        )
    
    # Improve the figure layout
    fig.update_layout(
        xaxis=dict(
            tickmode='array', 
            tickvals=list(range(year_range[0], year_range[1]+1))
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=80 if not compare_by_list or len(fig.data) < 8 else 40)
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Fix the hover template
    if not compare_by_list:
        fig.update_traces(
            hovertemplate='<b>Year</b>: %{x}<br><b>Salary</b>: $%{y:.2f}<extra></extra>'
        )
    else:
        # Use a different approach for the hover template
        for i, trace in enumerate(fig.data):
            category_name = trace.name
            # Add custom data for hover
            fig.data[i].customdata = [[category_name]] * len(trace.x)
            fig.data[i].hovertemplate = '<b>Year</b>: %{x}<br><b>Category</b>: %{customdata[0]}<br><b>Salary</b>: $%{y:.2f}<extra></extra>'
    
    # Calculate statistics
    stats = []
    if not filtered_data.empty:
        stats.append(html.H4("Statistics"))
        
        if not compare_by_list:
            overall_avg = filtered_data['Salary'].mean()
            min_salary = filtered_data['Salary'].min()
            max_salary = filtered_data['Salary'].max()
            
            stats.append(html.P(f"Overall Average: ${overall_avg:.2f}"))
            stats.append(html.P(f"Minimum: ${min_salary:.2f}"))
            stats.append(html.P(f"Maximum: ${max_salary:.2f}"))
            
            # Calculate growth rate
            if len(year_range) > 1:
                start_year_data = filtered_data[filtered_data['Year'] == year_range[0]]
                end_year_data = filtered_data[filtered_data['Year'] == year_range[1]]
                
                if not start_year_data.empty and not end_year_data.empty:
                    start_year = start_year_data['Salary'].mean()
                    end_year = end_year_data['Salary'].mean()
                    
                    if start_year > 0:
                        growth_rate = ((end_year - start_year) / start_year) * 100
                        stats.append(html.P(f"Growth Rate ({year_range[0]} to {year_range[1]}): {growth_rate:.2f}%"))
        else:
            # For combined comparisons
            if len(compare_by_list) == 1:
                compare_column = compare_by_list[0]
                column_header = compare_column
            else:
                compare_column = 'Combined'
                column_header = 'Categories'
            
            # Calculate statistics for each category
            stats_table = []
            headers = [column_header, "Average Salary", "Growth Rate"]
            
            # Create table rows
            rows = []
            categories = filtered_data[compare_column].unique()
            
            # Sort categories alphabetically (simple approach for combined categories)
            categories = sorted(categories)
            
            # Get the colors that Plotly assigned to each category
            category_colors = {}
            
            # Extract color information from the figure's data
            for trace in fig.data:
                category = trace.name
                color = trace.line.color
                category_colors[category] = color
            
            for category in categories:
                category_data = filtered_data[filtered_data[compare_column] == category]
                category_avg = category_data['Salary'].mean()
                
                # Calculate growth rate for this category
                growth_rate = "N/A"
                start_cat_data = category_data[category_data['Year'] == year_range[0]]
                end_cat_data = category_data[category_data['Year'] == year_range[1]]
                
                if not start_cat_data.empty and not end_cat_data.empty:
                    start_val = start_cat_data['Salary'].mean()
                    end_val = end_cat_data['Salary'].mean()
                    
                    if start_val > 0:
                        growth_rate = f"{((end_val - start_val) / start_val) * 100:.2f}%"
                
                # Use Plotly-assigned colors
                color = category_colors.get(category, '#000000')  # Default to black if not found
                row_style = {'border-left': f'5px solid {color}'}
                
                rows.append(html.Tr([
                    html.Td(category, style=row_style),
                    html.Td(f"${category_avg:.2f}"),
                    html.Td(growth_rate)
                ]))
            
            # Create the table with styling
            stats_table = dbc.Table([
                html.Thead(html.Tr([html.Th(h) for h in headers])),
                html.Tbody(rows)
            ], bordered=True, hover=True, striped=True)
            
            stats.append(stats_table)
    
    return fig, stats

# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    #app.save_html("dashboard1.html")
    #save_dash_html()