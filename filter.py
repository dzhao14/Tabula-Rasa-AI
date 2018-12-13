import numpy as np                                                              
import plotly.plotly as py                                                      
import plotly.graph_objs as go                                                  
import plotly.offline as offline

f = open('da', 'r')

lines = f.readlines() 
lines2 = []
for line in lines:
    if " loss: " in line:
        lines2.append(line)

vals = []
for line in lines2:
    sections = line.split('loss')
    v = sections[1].split()
    vals.append(float(v[1]))


layout = go.Layout(                                                         
        title="Training Loss",                                            
        yaxis=dict(                                                         
            title="Loss",                                                
            ),                                                              
        xaxis=dict(                                                         
            title="Training Games",                                              
            ),                                                              
        ) 

trace0 = go.Scatter(
        x=[i for i in range(len(vals))],
        y=vals,
        mode='lines',
        name="Training loss",
        )

fig = go.Figure(data=[trace0], layout=layout)
offline.plot(
        fig,
        image='png',
        filename="training_loss",
        )
