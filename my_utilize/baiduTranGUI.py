import tkinter as tk
import http.client
import hashlib
import urllib
import random
import json

class baiduTrans(object):
    def __init__(self):
        self.appid='20201217000649528'  # 填写你的appid
        self.secretKey = 'mOeGvrS1mSflQSHBUEaJ'  # 填写你的密钥
        self.httpClient = None
        self.myurl = '/api/trans/vip/translate'
        self.fromLang = 'auto'  # 原文语种
        self.toLang = 'zh'  # 译文语种
        self.salt = random.randint(32768, 65536)

    def trans(self,q):
        #q = 'The problem we have to address for training a unified detector is the ambiguity of predicted bounding boxes¯D that are not associated with any ground truth of the given image from dataset '
        sign = self.appid + q + str(self.salt) + self.secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl = self.myurl + '?appid=' + self.appid + '&q=' + urllib.parse.quote(
            q) + '&from=' + self.fromLang + '&to=' + self.toLang + '&salt=' + str(
            self.salt) + '&sign=' + sign
        output = ''
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)

            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)
            #import pprint
            # print (result)
            trains_result = result['trans_result']

            for i in range(len(trains_result)):
                output+=trains_result[i]['dst']+'\n'
                #print.pprint(trains_result[i])
                # print('src:', trains_result[i]['src'])
                # print('dst:', trains_result[i]['dst'])

        except Exception as e:
            print(e)
        finally:
            if httpClient:
                httpClient.close()
            return output

baiduTransAPI=baiduTrans()
#baiduTransAPI.trans('apple')

def trans():
    var = src.get('0.0','end')
    #print(var)
    output=baiduTransAPI.trans(var)
    dst.delete('0.0', 'end')
    dst.insert('end', output)


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