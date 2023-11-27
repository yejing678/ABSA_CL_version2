# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import spacy
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', add_special_tokens=True, do_lower_case=True)  # 区分大小写字母，

# nlp = spacy.load("en_core_web_sm")  # import spacy，用于分词
# sentence = nlp('CLS great food but the service was dreadful ! SEP')
text ='Our waiter was horrible; so rude and disinterested'
aspect = ['waiter']
tokens = tokenizer.tokenize(text)
sentence = ['[CLS]']
sentence.extend(tokens)
sentence.append('[SEP]')
# print(sentence)
# exit(0)
score = []

d = np.array(score)  # [17,1]
d = d.transpose()
col = [t for t in sentence]  # 需要显示的词
index = aspect  # 需要显示的词
df = pd.DataFrame(d, columns=col, index=index)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df, cmap='viridis', interpolation='nearest')
fig.colorbar(cax)

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fontdict = {'rotation': 45}  # 或者这样设置文字旋转
# Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
ax.set_yticklabels([''] + list(df.index))
plt.show()
