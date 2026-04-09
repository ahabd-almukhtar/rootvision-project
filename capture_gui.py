import sys, time
from PIL import ImageGrab
sys.path.insert(0, '/mnt/data/rootvision_project')
import tkinter as tk
from main import RootVisionApp

example = sys.argv[1]
outfile = sys.argv[2]

root = tk.Tk()
app = RootVisionApp(root)
app._load_example(example)
app.run_solver()
root.update()
root.after(1500, root.quit)
root.mainloop()
root.update()
# capture whole screen then crop window region
x = root.winfo_rootx(); y = root.winfo_rooty(); w = root.winfo_width(); h = root.winfo_height()
img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
img.save(outfile)
root.destroy()
