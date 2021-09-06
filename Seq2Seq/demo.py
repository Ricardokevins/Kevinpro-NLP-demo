import remi.gui as gui
from remi import start, App
from test import BotAPI

class MyApp(App):
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)

    def main(self):
        verticalContainer = gui.Container(width=1400, margin='0px auto', style={'display': 'block', 'overflow': 'hidden'})
        horizontalContainer = gui.Container(width='100%', layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='0px', style={'display': 'block', 'overflow': 'auto'})
        subContainer = gui.Container(height  = 650 , width=1300, style={'display': 'block', 'overflow': 'auto', 'text-align': 'center'})
        
        self.lbl = gui.Label('', width=1200, height=200, margin='10px')
        self.btInputDiag = gui.TextInput( width=1200, height=200, margin='10px')
        self.btInputDiag.onchange.do(self.on_input_dialog_confirm)

        self.bt = gui.Button('Press me to reset')
        self.bt.onclick.do(self.reset_event)
        #self.txt.onchange.do(self.on_text_area_change)
        # setting the listener for the onclick event of the button
       

        # appending a widget to another, the first argument is a string key
        subContainer.append([self.btInputDiag,self.lbl,self.bt])
        horizontalContainer.append([subContainer])

        menu = gui.Menu(width='100%', height='30px')
        m1 = gui.MenuItem('File', width=100, height=30)
       
        m11 = gui.MenuItem('Save', width=100, height=30)
        m12 = gui.MenuItem('Open', width=100, height=30)
        #m12.onclick.do(self.menu_open_clicked)
        m111 = gui.MenuItem('Save', width=100, height=30)
        #m111.onclick.do(self.menu_save_clicked)
        m112 = gui.MenuItem('Save as', width=100, height=30)
        #m112.onclick.do(self.menu_saveas_clicked)
        m3 = gui.MenuItem('Dialog', width=100, height=30)
       # m3.onclick.do(self.menu_dialog_clicked)

        menu.append([m1,  m3])
        m1.append([m11, m12])
        m11.append([m111, m112])

        menubar = gui.MenuBar(width='100%', height='30px')
        menubar.append(menu)

        verticalContainer.append([menubar, horizontalContainer])
        
        # returning the root widget
        return verticalContainer



    def on_input_dialog_confirm(self, widget, value):
        result = BotAPI(value)
        self.lbl.set_text(result)

    def reset_event(self,widget):
        self.lbl.set_text("")
        self.btInputDiag.set_value("")


# starts the web server
start(MyApp)
