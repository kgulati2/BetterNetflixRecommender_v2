import recommendation as rec
import PySimpleGUI as sg
from tkinter import *
from  tkinter import ttk

layout = [[sg.Text("Enter a Netflix Movie to Find Similar Options!")],
          [sg.Input()],
          [sg.Button('Submit')]]
window = sg.Window('BetterFlix', layout)
event, values = window.read()

#cleans whitespace before and after
values[0].strip()
#print('Vals after strip', values[0])

#List
rec_list = rec.recommend_movie(values[0])
s = '\n'.join([str(i) for i in rec_list])
sg.PopupScrolled("The Following Movies are Recommended:", f"{s}")

# testing
#print('Values', values[0])
#print('Rec actual', rec.recommend_movie("stranger thing"))

window.close()