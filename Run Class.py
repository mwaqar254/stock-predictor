import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler1=MinMaxScaler(feature_range=(0,1))



df_nse1 = pd.read_csv("INSTAGRAM.csv")

df_nse1["Date"]=pd.to_datetime(df_nse1.Date,format="%Y-%m-%d")
df_nse1.index=df_nse1['Date']


data1=df_nse1.sort_index(ascending=True,axis=0)
new_data1=pd.DataFrame(index=range(0,len(df_nse1)),columns=['Date','Close'])

for i in range(0,len(data1)):
    new_data1["Date"][i]=data1['Date'][i]
    new_data1["Close"][i]=data1["Close"][i]

new_data1.index=new_data1.Date
new_data1.drop("Date",axis=1,inplace=True)

dataset1=new_data1.values

train1=dataset1[100:1100,:]
valid1=dataset1[1100:2036,:]

scaler1=MinMaxScaler(feature_range=(0,1))
scaled_data1=scaler1.fit_transform(dataset1)

x_train1,y_train1=[],[]

for i in range(60,len(train1)):
    x_train1.append(scaled_data1[i-60:i,0])
    y_train1.append(scaled_data1[i,0])
    
x_train1,y_train1=np.array(x_train1),np.array(y_train1)

x_train1=np.reshape(x_train1,(x_train1.shape[0],x_train1.shape[1],1))

model=load_model("INSTA STOCK.h5")

inputs1=new_data1[len(new_data1)-len(valid1)-60:].values
inputs1=inputs1.reshape(-1,1)
inputs1=scaler1.transform(inputs1)

X_test1=[]
for i in range(60,inputs1.shape[0]):
    X_test1.append(inputs1[i-60:i,0])
X_test1=np.array(X_test1)

X_test1=np.reshape(X_test1,(X_test1.shape[0],X_test1.shape[1],1))
closing_price1=model.predict(X_test1)
closing_price1=scaler1.inverse_transform(closing_price1)

train1=new_data1[100:1100]
valid1=new_data1[1100:2036]
valid1['Predictions']=closing_price1






df_nse2 = pd.read_csv("TWITTER.csv")

df_nse2["Date"]=pd.to_datetime(df_nse2.Date,format="%Y-%m-%d")
df_nse2.index=df_nse2['Date']


data2=df_nse2.sort_index(ascending=True,axis=0)
new_data2=pd.DataFrame(index=range(0,len(df_nse2)),columns=['Date','Close'])

for j in range(0,len(data2)):
    new_data2["Date"][j]=data2['Date'][j]
    new_data2["Close"][j]=data2["Close"][j]

new_data2.index=new_data2.Date
new_data2.drop("Date",axis=1,inplace=True)

dataset2=new_data2.values

train2=dataset2[100:900,:]
valid2=dataset2[900:1609,:]

scaler2=MinMaxScaler(feature_range=(0,1))
scaled_data2=scaler2.fit_transform(dataset2)

x_train2,y_train2=[],[]

for j in range(60,len(train2)):
    x_train2.append(scaled_data2[j-60:j,0])
    y_train2.append(scaled_data2[j,0])
    
x_train2,y_train2=np.array(x_train2),np.array(y_train2)

x_train2=np.reshape(x_train2,(x_train2.shape[0],x_train2.shape[1],1))

model=load_model("TWITTER STOCK.h5")

inputs2=new_data2[len(new_data2)-len(valid2)-60:].values
inputs2=inputs2.reshape(-1,1)
inputs2=scaler2.transform(inputs2)

X_test2=[]
for j in range(60,inputs2.shape[0]):
    X_test2.append(inputs2[j-60:j,0])
X_test2=np.array(X_test2)

X_test2=np.reshape(X_test2,(X_test2.shape[0],X_test2.shape[1],1))
closing_price2=model.predict(X_test2)
closing_price2=scaler2.inverse_transform(closing_price2)

train2=new_data2[100:900]
valid2=new_data2[900:1609]
valid2['Predictions']=closing_price2



df= pd.read_csv("COMBINED.csv")


app.layout = html.Div([
   
    html.H1("STOCK PRICE ANALYSIS", style={"textAlign": "center", "background-color": "#ADD8E6"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='INSTAGRAM STOCK DATA',children=[
			html.Div([
				html.H2("ACTUAL CLOSING PRICE",style={"textAlign": "center","background-color": "#ADD8E6"}),
				dcc.Graph(
					id="Actual Data of instagram",
					figure={
						"data":[
							go.Scatter(
								x=train1.index,
								y=valid1["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Year'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("INSTAGRAM STOCK PREDICTED CLOSING PRICE",style={"textAlign": "center","background-color": "#ADD8E6" }),
				dcc.Graph(
					id="Predicted Data of Instagram",
					figure={
						"data":[
							go.Scatter(
								x=valid1.index,
								y=valid1["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Year'},
							yaxis={'title':'Closing Rate'}
						)
                        
                    
					}

				)				
			])        		


        ]),
        
        


    dcc.Tab(label='TWITTER STOCK DATA', children=[
            html.Div([
                html.H2("ACTUAL CLOSING PRICE",style={"textAlign": "center","background-color": "#ADD8E6"}),
				dcc.Graph(
					id="Actual Data of Twitter",
					figure={
						"data":[
							go.Scatter(
								x=train2.index,
								y=valid2["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Year'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("TWITTER STOCK PREDICTED CLOSING PRICE",style={"textAlign": "center","background-color": "#ADD8E6"}),
				dcc.Graph(
					id="Predicted Data of Twitter",
					figure={
						"data":[
							go.Scatter(
								x=valid2.index,
								y=valid2["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Year'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        
        dcc.Tab(label='OTHER STOCKS (H/L)', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center',"background-color": "#ADD8E6"}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center', "background-color": "#ADD8E6"}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])







@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)
