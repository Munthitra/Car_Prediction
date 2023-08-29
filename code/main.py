# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import math
import numpy as np
import pickle

#Import data file
# df = pd.read_csv(r'C:\Users\Munthitra\Desktop\Chaklam\a1\code\Cars (1).csv')
df = pd.read_csv('./Cars.csv')

#Name Cut
df['Car_Name'] = df['name'].str.split(" ").str[0]
df.drop(['name'], axis=1, inplace=True)

#Split Mileage, Engine, Max Power into value and unit
df[["Mileage_Value","Mileage_Unit"]] = df["mileage"].str.split(pat=' ', expand = True)
df[["Engine_Value","Engine_Unit"]] = df["engine"].str.split(pat=' ', expand = True)
df[["Max_Power_Value","Max_Power_Unit"]] = df["max_power"].str.split(pat=' ', expand = True)
df.drop(["mileage","engine","max_power"], axis=1, inplace=True)

#Remove LPG and CNG
new_df = df[df['fuel'].isin(['Diesel', 'Petrol'])]

#Remove Test drive car
new_df = new_df[new_df["owner"] != 'Test Drive Car']

#Change type to float
new_df['Mileage_Value'] = df['Mileage_Value'].str.split().str[0].astype(float)
new_df['Engine_Value'] = df['Engine_Value'].str.split().str[0].astype('float64')
new_df['Max_Power_Value'] = new_df['Max_Power_Value'].str.split().str[0].astype('float64')

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            dbc.Label("Max_Power_Value"),
            dbc.Input(id="Max_Power_Value", type="number", placeholder="Put a value of Maximum Power in bhp"),
            dbc.Label("Mileage_Value"),
            dbc.Input(id="Mileage_Value", type="number", placeholder="Put a value of Mileage in km/l"),
            dbc.Label("km_driven"),
            dbc.Input(id="km_driven", type="number", placeholder="Put a value for Kilometers driven in km"),
            dbc.Button(id="submit", children="calculate y", color="primary", className="me-1"),
            dbc.Label("y is: "),
            html.Output(id="y", children="")
        ],
        className="mb-3")
    ])

], fluid=True)

#Callback the Input
@callback(
    Output(component_id="y", component_property="children"),
    State(component_id="Max_Power_Value", component_property="value"),
    State(component_id="Mileage_Value", component_property="value"),
    State(component_id="km_driven", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)

def prediction (Max_Power_Value,Mileage_Value,km_driven,submit):
    if Max_Power_Value == None:
        Max_Power_Value = df["Max_Power_Value"].median()
    if Mileage_Value == None:
        Mileage_Value = df["Mileage_Value"].mean()
    if km_driven == None:
        km_driven = df["km_driven"].median()
    model = pickle.load(open("/root/code/124022 car_prediction.model", 'rb')) # Import model
    sample = np.array([[Max_Power_Value, Mileage_Value, math.log(km_driven)]]) 
    result = np.exp(model.predict(sample)) #Predict price
    return f"The predictive car price is {int(result[0])}"
  
# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)