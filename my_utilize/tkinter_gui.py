import tkinter as tk

def trans():
    var = src.get('0.0','end')
    #print(var)
    dst.delete('0.0', 'end')
    dst.insert('end', var)

window=tk.Tk()
window.title('Translation from en to zh')
window.geometry('500x300')


src=tk.Text(window,height=3)
dst=tk.Text(window,height=3)

l_head = tk.Label(window, text='Translator', bg='green', font=('Arial', 12), width=30, height=2)
l_src = tk.Label(window, text='原文', bg='white',foreground='red',font=('Arial', 12), width=20, height=1)
l_dst = tk.Label(window, text='结果',  bg='white',foreground='blue',font=('Arial', 12), width=20, height=1)
b = tk.Button(window, text='翻译', font=('Arial', 12), width=10, height=1, command=trans)

l_head.pack()
l_src.pack()
src.pack()
l_dst.pack()
dst.pack()
b.pack()

window.mainloop()