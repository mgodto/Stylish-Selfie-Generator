import sys 
from Demo_StyletransferAPP import *
#sys.setrecursionlimit(1000000) 
if __name__ == '__main__':
    def toggleFS(e):
        if window.tk.attributes('-fullscreen'):
            window.tk.attributes('-fullscreen', False)
        else:
            window.tk.attributes('-fullscreen', True)
        
    window = StyletransferApp()
    window.EnableButton()
    window.tk.bind("f", toggleFS)
    window.tk.mainloop()