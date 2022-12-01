import networkx as nx
import nltk
import pandas as pd
import numpy as np
import string
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import preprocessing
g=nx.Graph()
tweet=[]
label=[]
df = pd.read_excel(r"/content/shufflefilenew2.xlsx")
#print(str(df.values[0][1]))
tweetcred=np.zeros((len(df),1))
for i in range(0,len(df)):
    filee=open('/content/nontrue15162/'+str(df.values[i][1]),'r')
    g.add_node(int(str(df.values[i][1])[0:-4]))
    labelfile=open('/content/labelf.txt','r')
    source_tweet=open('/content/source_tweetsf.txt','r')
    for lf in labelfile:
        if(str(df.values[i][1])[0:-4] in lf):
            if('non-rumor'in lf):
                #endchar=lf.find(":")
                #label.append(lf[0:endchar])
                label.append('non-rumor')
                break
            else:
                label.append('rumor')
                break   
    for st in source_tweet:
        if(str(df.values[i][1])[0:-4] in st):
             startchar=st.find('\t',1,20)
             tweet.append(st[startchar+1:])
             break       
    #tweetattribute
    
    text =tweet[i]
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    tokens_without_sw=[word for word in tokens_without_sw if not word in string.punctuation]
    a=nltk.FreqDist(word.lower() for word in tokens_without_sw)
    sumscore=0
    for word in text_tokens:
        wordd=TextBlob(word)
        score = wordd.sentiment.polarity
        #print(word,":",score)
        sumscore+=score
    g.nodes[int(str(df.values[i][1])[0:-4])]['id'] =int(str(df.values[i][1])[0:-4])
    g.nodes[int(str(df.values[i][1])[0:-4])]['tokens']=len(tokens_without_sw)
    #g.nodes[int(str(df.values[i][1])[0:-4])]['retweet'] =len(filee)
    g.nodes[int(str(df.values[i][1])[0:-4])]['label'] =label[i]
    #print(g.nodes[int(str(df.values[i][1])[0:-4])]['retweet'])
    #hashtagattribute
    h1=''
    h2=''
    h3=''
    h4=''
    h5=''
    index1=text.find('#')
    index11=text.find(':',index1)
    index12=text.find(' ',index1)
    h1=(text[index1:index12].upper())
        #g.nodes[text[index1:index12].upper()]['popularity']=df.values[i][11]
        
    index2=text.find('#',index12)
    index22=text.find(' ',index2)
    h2=(text[index2:index22].upper())
        #g.nodes[text[index2:index22].upper()]['popularity']=df.values[i][12]
    index3=text.find('#',index22)
    index32=text.find(' ',index3)
    h3=(text[index3:index32].upper())
        #g.nodes[text[index3:index32].upper()]['popularity']=df.values[i][13]
    index4=text.find('#',index32)
    index42=text.find(' ',index4)
    h4=(text[index4:index42].upper())
        #g.nodes[text[index4:index42].upper()]['popularity']=df.values[i][14]
    
    index5=text.find('#',index42)
    index52=text.find(' ',index5)
    h5=(text[index5:index52].upper())
        #g.nodes[text[index5:index52].upper()]['popularity']=df.values[i][15]

    lin=filee.readline()
    pindex1 = lin.find('>',15,30)
    pindex2 = lin.find(',',pindex1,pindex1+15)
    pname=lin[pindex1+3:pindex2-1]
    for j in range(i+1,len(df)):
      fileee=open('/content/nontrue15162/'+str(df.values[j][1]),'r')
      lineee=fileee.readline()
      pindex1 = lineee.find('>',15,30)
      pindex2 = lineee.find(',',pindex1,pindex1+15)
      pname2=lineee[pindex1+3:pindex2-1]
      tweet2=''
      for st2 in source_tweet:
        if(str(df.values[j][1])[0:-4] in st2):
             startchar=st2.find('\t',1,20)
             tweet2=(st2[startchar+1:])
             break
      if(pname in pname2):
          g.add_edge(int(str(df.values[i][1])[0:-4]),int(str(df.values[j][1])[0:-4]),weight=0) 
      elif((h1 in tweet2 and h1 !='') or (h2 in tweet2 and h2 !='')or (h3 in tweet2 and h3 !='') or (h4 in tweet2 and h4 !='')or (h5 in tweet2 and h5 !='')):
          g.add_edge(int(str(df.values[i][1])[0:-4]),int(str(df.values[j][1])[0:-4]),weight=0) 
#print("geraphadd")
    num=0
    for line in filee:
        pindex1 = line.find('[',0,3)
        pindex2 = line.find(',',0,18)
        cindex1 = line.find('>',20,60)
        cindex2 = line.find(',',40,70)
        if( line.find(',',80,105)>80):
            ctimeindex1=line.find(',',80,110)
        else:
            ctimeindex1=line.find(',',65,85)
        ctimeindex2=line.find(']',65,130)
        ptimeindex1=line.find(',',25,40)
        ptimeindex2=line.find(']',ptimeindex1,50)
        pname=line[pindex1+2:pindex2-1]
        cname=line[ cindex1+3: cindex2-1]
        ctime=line[ ctimeindex1+3: ctimeindex2-1]
        ptime=line[ ptimeindex1+3: ptimeindex2-1]
        if(str(pname) in lin):
            #g.add_edge(df.values[i][1],cname,weight=round(float(ctime)))
            g.add_edge(int(str(df.values[i][1])[0:-4]),cname,weight=round(float(ctime)))
            g.nodes[cname]['label']='unlabel'
            num=num+1
        if(num>=2):
          break;    
    print(i,"add",int(str(df.values[i][1])[0:-4]),g.degree[int(str(df.values[i][1])[0:-4])])           
#print(g.adj[str(407159686786732032)])    
print(g.number_of_nodes())
X=[]
label=[]
centrality=[]
centrality.append(['id','label','degree','degreecent','closeness_centrality','pagerank'])

for node in range(600,602):
     centrality.append([df.values[node][1],g.nodes[int(str(df.values[node][1])[0:-4])]['label'],g.degree[int(str(df.values[node][1])[0:-4])],nx.degree_centrality(g)[int(str(df.values[node][1])[0:-4])],nx.closeness_centrality(g,u=None, distance='weight', wf_improved=True)[int(str(df.values[node][1])[0:-4])],nx.pagerank(g,weight='weight', max_iter=200,alpha=0.9)[int(str(df.values[node][1])[0:-4])]])
     print('centrality', node ,' add',centrality[-1])

centrality2=np.array(centrality)
df3 = pd.DataFrame(centrality2)
df3.to_excel(r'/content/new700.xlsx','Sheet1')


centrality=[]
centrality.append(['id','label','degree','degreecent','closeness_centrality','pagerank'])

for node in range(700,800):
    centrality.append([df.values[node][1],g.nodes[int(str(df.values[node][1])[0:-4])]['label'],g.degree[int(str(df.values[node][1])[0:-4])],nx.degree_centrality(g)[int(str(df.values[node][1])[0:-4])],nx.closeness_centrality(g,u=None, distance='weight', wf_improved=True)[int(str(df.values[node][1])[0:-4])],nx.pagerank(g,weight='weight', max_iter=200,alpha=0.9)[int(str(df.values[node][1])[0:-4])]])
    print('centrality', node ,' add',centrality[-1])

centrality2=np.array(centrality)
df3 = pd.DataFrame(centrality2)
df3.to_excel(r'/content/new800.xlsx','Sheet1')

centrality=[]
centrality.append(['id','label','degree','degreecent','closeness_centrality','pagerank'])

for node in range(800,900):
    centrality.append([df.values[node][1],g.nodes[int(str(df.values[node][1])[0:-4])]['label'],g.degree[int(str(df.values[node][1])[0:-4])],nx.degree_centrality(g)[int(str(df.values[node][1])[0:-4])],nx.closeness_centrality(g,u=None, distance='weight', wf_improved=True)[int(str(df.values[node][1])[0:-4])],nx.pagerank(g,weight='weight', max_iter=200,alpha=0.9)[int(str(df.values[node][1])[0:-4])]])
    print('centrality', node ,' add',centrality[-1])
    
centrality2=np.array(centrality)
df3 = pd.DataFrame(centrality2)
df3.to_excel(r'/content/new900.xlsx','Sheet1')
