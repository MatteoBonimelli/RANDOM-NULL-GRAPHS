class grafo_r:
    def __init__(self):
        self._nodi=dict()                #creo un dizionario di nodi con valori i nodi adiacenti
        self._col=dict()                 #creo un dizionario aseggnando a ogni nodo il proprio colore    
    
    def add_nodo(self,nodo,col=""):    
        if nodo not in self._nodi:       #funzione che aggiunge un nodo nei due dizionari iniziali
            self._nodi[nodo]=[]
            self._col[nodo]=col
    def add_arco(self,nodo1,nodo2):
        if nodo1 not in self._nodi:
            self.add_nodo(nodo1)
        if nodo2 not in self._nodi:         #aggiunge i nodi adiacenti in input al dizionario _nodi
            self.add_nodo(nodo2)
        if nodo2 not in self._nodi[nodo1]:
            self._nodi[nodo1].append(nodo2)
        if nodo1 not in self._nodi[nodo2]:
            self._nodi[nodo2].append(nodo1)
    def nodo_exist(self,nodo):
        return nodo in self._nodi      #verifica se il nodo esiste
    def arco_exist(self,nodo1,nodo2):
        return (nodo1 in self._nodi[nodo2] and nodo1 in self._nodi[nodo2])  #verifica se l'arco esiste
    def numero_nodi(self):
        return len(self._nodi)   #restituisce il numero di nodi
    def numero_archi(self):
        a=0
        for n in self._nodi:
            a+=len(self._nodi[n])   #restituisce il numero di archi
        return a//2
    def lista_nodi(self):
        return list(self._nodi.keys())  #lista dei nodi del grafo
    def lista_archi(self):
        archi=[]
        for n in self._nodi:
            for a in self._nodi[n]:  #lista di archi senza duplicati
                if [a,n] not in archi:
                    archi.append([n,a])
        return archi
    def lista_archi_col(self):
        archi=[]
        numero=[]
        for n in self._nodi:               #lista di archi rappresentati per colore
            for a in self._nodi[n]:
                if [a,n] not in numero:
                    numero.append([n,a])
                    archi.append([self._col[n],self._col[a]])
        return archi
    def lista_gradi(self):
        gradi=[]
        for n in self._nodi:
            gradi.append(len(self._nodi[n])) #da una lista dei gradi di tutti i nodi
        return gradi
    def dict_gradi(self):
        gradi={}
        for n in self._nodi:
            gradi[n]=len(self._nodi[n]) #dizionario che assegna a ogni nodo il proprio grado
        return gradi
    def add_col(self,n,col):
        self._col[n]=col    #aggiunge o modifica il colore di un certo nodo
    def freq_col(self):
        from collections import Counter
        return Counter(list(self._col.values()))  #
    def omo(self):
        archi=[]
        archi_omo=0
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi:
                    archi.append([n,a])
                    if self._col[n]==self._col[a]: #restituisce il numero di archi omofili
                        archi_omo+=1
        return archi_omo
    
            
    def archi_omo(self):
        archi=[]
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi: 
                    if self._col[n]==self._col[a]: #restituisce la lista degli archi omofili
                        archi.append([n,a])
                        
        return archi
    def omo_by_col(self):
        omofili=dict()
        colori = sorted(list(set(self._col.values())))
        archi_col=self.lista_archi_col()
        for c in colori:
            count=0
            for a in range(len(archi_col)):
                if archi_col[a][0]==c and archi_col[a][1]==c: #dizionario numero archi omofili per colore
                    count+=1
            omofili[c]=count
        return omofili
            
    def ete(self):
        archi=[]
        archi_ete=0
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi:
                    archi.append([n,a])
                    if self._col[n]!=self._col[a]: #numero di archi eterofili
                        archi_ete+=1
        return archi_ete
    def archi_ete(self):
        archi=[]
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi: 
                    if self._col[n]!=self._col[a]: #lista archi eterofili
                        archi.append([n,a])
                        
        return archi
    def ete_by_col(self):
        eterofili=dict()
        colori = sorted(list(set(self._col.values())))
        archi_col=self.lista_archi_col()
        for c1 in colori:
            for c2 in colori:
                if c1!=c2:
                    count=0
                    for a in range(len(archi_col)):
                        if (archi_col[a][0]==c1 and archi_col[a][1]==c2)or(archi_col[a][0]==c2 and archi_col[a][1]==c1):
                            count+=1
                    if (c2,c1) not in eterofili.keys():   #dizionario numero archi eterofili per colore
                        eterofili[(c1,c2)]=count
        return eterofili
        
    def random_coloring(self,ripetizioni=1):
        import random
        diz_rip=dict()
        nodi=list(self._col.keys())
        colori=list(self._col.values())
        for rip in range(ripetizioni):
            new_col=dict()
            colori_mix=random.sample(colori,k=len(colori)) #CREA R RIPETIZIONI DEL RANDOM COLORING DEL DIZIONARIO NODO COLORE
            for nodo in nodi:
                new_col[nodo]=colori_mix[nodi.index(nodo)]
            diz_rip[rip]=new_col
        return diz_rip
    def configuration(self,semp=True):
        import random
        gradi=self.lista_gradi()
        mezzi_archi=[]
        nodi=list(self._nodi.keys())
        for nodo in nodi:
            mezzi_archi.extend([nodo]*gradi[nodi.index(nodo)])
        if len(mezzi_archi) % 2:
            raise ValueError('Somma gradi dispari')
        mezzi_archi_r=random.sample(mezzi_archi,k=len(mezzi_archi)) #CONFIGURATION MANTENENDO I GRADI DEI NODI
        archi=[]                                                    #CON SEMP=TRUE LEVA GLI ARCHI DUPLICATI E I NODI
        for i in range (0,len(mezzi_archi_r),2):                    #ADIACENTI A SE STESSI
            archi.append([mezzi_archi_r[i],mezzi_archi_r[i+1]])
        archi_semp=archi
        if semp:
            archi_rem=0
            for i in archi:
                if i[0]==i[1] or [i[1],i[0]]in archi or archi.count(i)>=2 :
                    archi_semp.remove(i)
                    archi_rem+=1
        if semp:
            return archi_semp,{'archi rimossi con semplificazione':archi_rem}
        return archi_semp
    def configuration_sw(self,ripetizioni=10000):
        import random
        residui=self.dict_gradi()
        mezzi_archi=[]
        nodi=list(self._nodi.keys())
        for nodo in nodi:
            mezzi_archi.extend([nodo]*residui[nodo])
        if len(mezzi_archi) % 2:
            raise ValueError('Somma gradi dispari')     #CONFIGURATION STUB WEIGHTED
        archi_storti=[]
        for r in range(ripetizioni):
            residui=self.dict_gradi()
            mezzi_archi_r=random.sample(mezzi_archi,k=len(mezzi_archi))
            archi=[]
            risolto=True
            
            while mezzi_archi_r:
                n1=mezzi_archi_r.pop()
                if residui[n1]>=1:
                    residui[n1]-=1
                    pesi=[residui[p] for p in mezzi_archi_r]
                    n2=random.choices(mezzi_archi_r,weights=pesi,k=1)[0]
                    residui[n2]-=1
                    mezzi_archi_r.remove(n2)
                    if n1==n2 or [n2,n1]in archi or archi.count([n1,n2])>=2 :
                        risolto=False
                        archi_storti.append([n1,n2])
                        break
                    archi.append([n1,n2])
            if risolto:
                return archi
        
        raise RuntimeError(f"Impossibile ottenere un grafo semplice con {ripetizioni} tentativi.")
    def set_RC(self):
        prova=self.random_coloring() #FISSA UN RANDOM COLORING
        return prova
    def set_CM(self):
        test=self.configuration_sw() #FISSA UN CONFIGURATION MODEL
        return test
    def V_A_RC(self):
        m=len(self.lista_archi())
        n=len(self.lista_nodi())
        c=self.freq_col()
        mu={}
        for c1 in list(c.keys()):
            for c2 in list(c.keys()):
                if c1==c2:
                    mu[(c1,c2)]=m*(c[c1]*(c[c1]-1))/(n*(n-1)) #VALORE ATTESO DEL RANDOM COLORING
                elif c1<c2:
                    mu[(c1,c2)]=2*m*(c[c1]*c[c2])/(n*(n-1))
        return mu
    def V_A_CM(self):#VALORE ATTESO CM
        m=len(self.lista_archi())
        gradi=self.dict_gradi()
        d={}
        c=self.freq_col()
        for col in list(c.keys()):
            d[col]=0
        for g in list(gradi.keys()):
            d[self._col[g]]+=gradi[g]
        mu={}
        for d1 in list(d.keys()):
            for d2 in list(d.keys()):
                if d1==d2:
                    mu[(d1,d2)]=m*(d[d1]/(2*m))**2
                if d1<d2:
                    mu[(d1,d2)]=d[d1]*d[d2]/(2*m)
                    
        return mu
            
    
        
    def mu_RC_montecarlo(self,ripetizioni=1000,varianza=False): #GENERA R RIPETIZIONI MONTECARLO E CALCOLA LA MEDIA 
        mu_archi=dict()                                         #PER SIMULAZIONI RC
        prove=dict()                                            #var=True restituisce anche la varianza
        c=self.freq_col()
        col=list(c.keys())
        for i in range(ripetizioni) :
            prova=self.random_coloring()
            prove[i]=prova[0]
            archi=dict()
            for a in self.lista_archi():
                col1=prove[i][a[0]]
                col2=prove[i][a[1]]
                if col1>col2:
                    col1,col2=col2,col1
                archi[(col1,col2)]=archi.get((col1,col2),0)+1
            mu_archi[i]=archi
        mu=dict()
        valori=mu_archi.values()
        for c1 in col:
            for c2 in col:
                if c1>c2:
                    None
                else:
                    mu[(c1,c2)]=mu.get((c1,c2), 0)+sum(p.get((c1,c2),0) for p in valori)/ripetizioni
        if varianza:
            var=dict()
            for c1 in col:
                for c2 in col:
                    if c1>c2:
                        None
                    else:
                        var[(c1,c2)]=var.get((c1,c2), 0)+sum((p.get((c1,c2),0)-mu[c1,c2])**2 for p in valori)/(ripetizioni-1)
        
            return mu,var
        return mu
    
    def diff_mu_RC(self):
        val_att=self.V_A_RC()
        media=self.mu_RC_montecarlo()
        diff=dict()
        for a in val_att:
                diff[a]=val_att[a]-media[a] #diff tra valore atteso e media delle simulazioni montecarlo
        return diff
    def diff_mu_perc_RC(self,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.V_A_RC()
        media=self.mu_RC_montecarlo(ripetizioni) #differenza RC IN PERCENTUALE
        diff=dict()
        for a in val_att:
                diff[a]=(val_att[a]-media[a])/val_att[a]
        return diff,sum(diff.values())/len(diff.values())
    
    def media_diff_mu_RC(self):
        diff=self.diff_mu_RC()
        media=sum(list(diff.values()))/len(list(diff.values())) #MEDIA DI TUTTE LE DIFFERENZE
        return media
    def errore_mu_RC(self,relativo=False,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.V_A_RC()
        media=self.mu_RC_montecarlo(ripetizioni=ripetizioni)#ERRORE RC CON RELATIVO=TRUE DA L'ERRORE RELATIVO
        err=dict()
        err_omo=dict()
        err_ete=dict()
        for a in val_att:
            
            if relativo:
                valore=abs(val_att[a]-media[a])/val_att[a]
            else:
                valore=abs(val_att[a]-media[a])
            err[a]=valore
            if a[0]==a[1]:
                err_omo[a]=valore
            else:
                err_ete[a]=valore
        return err,err_omo,err_ete
    def media_err_RC(self,ripetizioni=1000):
        err=self.errore_mu_RC(True,ripetizioni)[0] #MEDIA DI TUTTI GLI ERRORI RELATIVI
        media=sum(err.values())/len(err.values())
        return media
    def plot_err_rip_RC(self):
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [self.media_err_RC(r) for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2) #GRAFICO MEDIA ERRORI RELATIVI RC PER NUMERO DI SIMULAZIONI MONTECARLO
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Errore relativo medio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_diff_rip_RC(self):
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [round(self.diff_mu_perc_RC(r)[1],10) for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2) #GRAFICO DIFFERENZE PERCENTUALE RC PER SIMULAZIONI RC
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Scarto relativo medio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    def var_RC_montecarlo(self,ripetizioni=1000):  #VAR R SIMULAZIONI MONTECARLO RC
        mu_archi=dict()
        prove=dict()
        c=self.freq_col()
        col=list(c.keys())
        for i in range(ripetizioni) :
            prova=self.random_coloring()
            prove[i]=prova[0]
            archi=dict()
            for a in self.lista_archi():
                col1=prove[i][a[0]]
                col2=prove[i][a[1]]
                if col1>col2:
                    col1,col2=col2,col1
                archi[(col1,col2)]=archi.get((col1,col2),0)+1
            mu_archi[i]=archi
        mu=dict()
        valori=mu_archi.values()
        for c1 in col:
            for c2 in col:
                if c1>c2:
                    None
                else:
                    mu[(c1,c2)]=sum(p.get((c1,c2),0) for p in valori)/ripetizioni
        var=dict()
        for c1 in col:
            for c2 in col:
                if c1>c2:
                    None
                else:
                    var[(c1,c2)]=sum((p.get((c1,c2),0)-mu[c1,c2])**2 for p in valori)/(ripetizioni-1)
        
        return var
    
    
    
    
    def mu_CM_montecarlo(self,ripetizioni=1000,varianza=False): #MEDIE R SIMULAZIONI MONTECARLO CM
        mu_archi=dict()
        prove=dict()
        c=self.freq_col()
        col=list(c.keys())
        for i in range(ripetizioni) :
            prove[i]=self.configuration(semp=False)
            archi=dict()
            for a in prove[i]:
                col1=self._col[a[0]]
                col2=self._col[a[1]]
                if col1>col2:
                    col1,col2=col2,col1
                archi[(col1,col2)]=archi.get((col1,col2),0)+1
            for colore1 in col:
                for colore2 in col:
                    if colore1>colore2:
                        colore1,colore2=colore2,colore1
                    archi[(col1,col2)]=archi.get((col1,col2),0)
            mu_archi[i]=archi
        mu=dict()
        valori=mu_archi.values()
        for c1 in col:
            for c2 in col:
                if c1>c2:
                    None
                else:
                    mu[(c1,c2)]=(sum(p.get((c1,c2),0) for p in valori)/ripetizioni)
        if varianza:
            var=dict()
            for c1 in col:
                for c2 in col:
                    if c1>c2:
                        None
                    else:
                        var[(c1,c2)]=var.get((c1,c2), 0)+sum((p.get((c1,c2),0)-mu[c1,c2])**2 for p in valori)/(ripetizioni-1)
        
            return mu,var
       
        return mu
    
    
    def diff_mu_CM(self,ripetizioni=1000): #DIFFERENZE VALORE ATTESO MENO MEDIE MONTECARLO PER IL CM
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.V_A_CM()
        media=self.mu_CM_montecarlo(ripetizioni)
        diff=dict()
        diff_omo=dict()
        diff_ete=dict()
        for a in val_att:
                diff[a]=(val_att[a]-media[a])
                if a[0]==a[1]:
                    diff_omo[a]=val_att[a]-media[a]
                else:
                    diff_ete[a]=val_att[a]-media[a]
        return diff,diff_omo,diff_ete
    def diff_mu_perc_CM(self,ripetizioni=1000): #DIFFERENZE IN PERCENTUALE CM
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.V_A_CM()
        media=self.mu_CM_montecarlo(ripetizioni)
        diff=dict()
        for a in val_att:
                diff[a]=(val_att[a]-media[a])/val_att[a]
                
        return diff,sum(diff.values())/len(diff.values())
    def errore_mu_CM(self,relativo=False,ripetizioni=1000): #ERRORE CM
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.V_A_CM()
        media=self.mu_CM_montecarlo(ripetizioni=ripetizioni)
        err=dict()
        err_omo=dict()
        err_ete=dict()
        for a in val_att:
            
            if relativo:
                valore=abs(val_att[a]-media[a])/val_att[a]
            else:
                valore=abs(val_att[a]-media[a])
            err[a]=valore
            if a[0]==a[1]:
                err_omo[a]=valore
            else:
                err_ete[a]=valore
        return err,err_omo,err_ete
    def media_err_CM(self,ripetizioni=1000):      #MEDIA ERRORE RELATIVO CM
        err=self.errore_mu_CM(True,ripetizioni)[0]
        media=sum(err.values())/len(err.values())
        return media
        
    
    def media_diff_mu_CM(self,ripetizioni=1000): #MEDIA DIFFERENZE CM
        diff=self.diff_mu_CM(ripetizioni)[0]
        diff_omo=self.diff_mu_CM()[1]
        diff_ete=self.diff_mu_CM()[2]
        lista_diff=list(diff.values())
        lista_omo=list(diff_omo.values())
        lista_ete=list(diff_ete.values())
        media=round(sum(lista_diff)/len(lista_diff),18)
        media_abs=round(sum([abs(lista_diff[i])for i in range(len(lista_diff))])/len(lista_diff),5)
        media_omo=round(sum(lista_omo)/len(lista_omo),5)
        media_ete=round(sum(lista_ete)/len(lista_ete),5)
        
        return media,media_omo,media_ete,media_abs
    def z_score_CM(self,omo=False,ete=False): #(VALORE ATTESO - MEDIE MONTECARLO)/VARIANZA MONTECARLO CM
        from math import sqrt
        val_att=self.V_A_CM()
        prove=self.mu_CM_montecarlo(ripetizioni=100,varianza=True)
        media=prove[0]
        var=prove[1]
        z=dict()
        z_omo=dict()
        z_ete=dict()
        for a in val_att:
                z[a]=(media[a]-val_att[a])/sqrt(var[a])
                if a[0]==a[1]:
                    z_omo[a]=(media[a]-val_att[a])/sqrt(var[a])
                else:
                    z_ete[a]=(media[a]-val_att[a])/sqrt(var[a])
        if omo and ete:
            return z_omo,z_ete
        elif omo:
            return z_omo
        elif ete:
            z_ete
            
        return z
    def scatter_val_att_vs_montecarlo_CM(self,ripetizioni=1000): #SCATTER PLOT VALORE ATTESO VS MEDIE MONTECARLO
        import matplotlib.pyplot as plt
        val_att=self.V_A_CM()
        media=self.mu_CM_montecarlo(ripetizioni)
        col = list(val_att.keys())
        x = ([val_att[c] for c in col])
        y = ([media[c] for c in col])
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        plt.figure(figsize=(7,6))
        for c in col:
            plt.scatter(val_att[c], media[c],
            color="steelblue",
            s=80,
            edgecolor="black",
            alpha=0.8)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.xlabel("Valore atteso")
        plt.ylabel("Media Monte Carlo")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
    def scatter_val_att_vs_montecarlo_RC(self):#SCATTER PLOT VALORE ATTESO VS MEDIE MONTECARLO RC
        import matplotlib.pyplot as plt
        val_att=self.V_A_RC()
        media=self.mu_RC_montecarlo()
        col = list(val_att.keys())
        x = ([val_att[c] for c in col])
        y = ([media[c] for c in col])
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        plt.figure(figsize=(7,6))
        for c in col:
            plt.scatter(val_att[c], media[c],
            color="steelblue",
            s=80,
            edgecolor="black",
            alpha=0.8)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.xlabel("Valore atteso")
        plt.ylabel("Media Monte Carlo")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

   
            

        

    def IC_CM(self): #CREA INTERVALLI DI CONFIDENZA CON LE MEDIE E VARIANZE MONTECARLO E RESTITUISCE SE E QUANTI VALORI ATTESI SONO INCLUSI NEGLI INTERVALLI
        from math import sqrt
        from scipy.stats import norm
        R =1000
        monte=self.mu_CM_montecarlo(ripetizioni=R,varianza=True)
        mu=monte[0]
        sigma=monte[1]
        z = float(norm.ppf(0.975))
        IC=dict()
        for c in mu:
            IC_L=mu[c]-z*sqrt(sigma[c])/sqrt(R)
            IC_H=mu[c]+z*sqrt(sigma[c])/sqrt(R)
            IC[c]=[IC_L,IC_H]
        v_a=self.V_A_CM()
        in_int=dict()
        for co in v_a:
            in_int[co]=v_a[co]>IC[co][0] and v_a[co]<IC[co][1]
            
        return IC,in_int,sum(in_int.values())/len((in_int.values()))*100
    def plot_err_rip_CM(self): #PLOT MEDIA ERRORE CM PER SIMULAZIONI CM
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [self.media_err_CM(r) for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2)
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Errore relativo medio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_diff_rip_CM(self): #PLOT MEDIA DIFFERENZE PERCENTUALI PER NUMERO SIMULAZIONI CM
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [round(self.diff_mu_perc_CM(r)[1],10) for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2)
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Scarto relativo medio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
        
        
        
            
    def var_CM_montecarlo(self,ripetizioni=1000): #VARIANZA MONTECARLO CON R SIMULAZIONI CM
        mu_archi=dict()
        prove=dict()
        c=self.freq_col()
        col=list(c.keys())
        for i in range(ripetizioni) :
            prove[i]=self.configuration(semp=False)
            archi=dict()
            for a in prove[i]:
                col1=self._col[a[0]]
                col2=self._col[a[1]]
                if col1>col2:
                    col1,col2=col2,col1
                archi[(col1,col2)]=archi.get((col1,col2),0)+1
            mu_archi[i]=archi
        mu=dict()
        valori=mu_archi.values()
        for c1 in col:
            for c2 in col:
                if c1>c2:
                    None
                else:
                    mu[(c1,c2)]=mu.get((c1,c2), 0)+(sum(p.get((c1,c2),0) for p in valori)/(ripetizioni-1))
        var=dict()
        for c1 in col:
            for c2 in col:
                if c1>c2:
                    None
                else:
                    var[(c1,c2)]=var.get((c1,c2), 0)+sum((p.get((c1,c2),0)-mu[c1,c2])**2 for p in valori)/(ripetizioni-1)
        
        return var
    def var_CM(self): #VARIANZA APPROSSIMATA CM
        mu=self.V_A_CM()
        m=len(self.lista_archi())
        var=dict()
        for a in mu:
            var[a]=mu[a]*(1-mu[a]/(2*m))
        return var
    def rap_var_CM(self,ripetizioni=1000): #RAPPORTI VARIANZA APPROSSIMATA CM/VARIANZA MONTECARLO CM
        rap=dict()
        var=self.var_CM()
        monte=self.var_CM_montecarlo(ripetizioni)
        for a in var:
            rap[a]=var[a]/monte[a]
        return rap,sum(rap.values())/len(rap.values())
    def err_rel_var_CM(self,ripetizioni=1000): #ERRORE RELATIVO VARIANZE CM
        err=dict()
        var=self.var_CM()
        monte=self.var_CM_montecarlo(ripetizioni)
        for a in var:
            err[a]=abs(var[a]-monte[a])/monte[a]
        return err,sum(err.values())/len(err.values())
    def plot_err_var_CM(self): #GRAFICO ERRORE RELATIVO VARIANZE PER SIMULAZIONI CM
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [self.err_rel_var_CM(r)[1] for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2)
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Errore relativo medio varianza")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def plot_rap_var_CM(self):#RAPPORTO VARIANZE PER SIMULAZIONI CM
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [self.rap_var_CM(r)[1] for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2)
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Rapporto varianza medio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def fall(self,x,k): #FUNZIONE FALL
        ris=1
        for i in range(k):
            ris*=(x-i)
        return ris
    def pi3(self): #PI GRECO 3 NUMERO CAMMINI LUNGHEZZA 2
        from math import comb
        pi3=sum(comb(k,2) for k in self.lista_gradi() if k>=2)
        return pi3
    def var_RC_esatta(self): #VARIANZA RC TEORICA
        from math import comb
        m=len(self.lista_archi())
        n=len(self.lista_nodi())
        c=self.freq_col()
        mu={}
        for c1 in list(c.keys()):
            for c2 in list(c.keys()):
                if c1==c2:
                    mu[(c1,c2)]=m*(c[c1]*(c[c1]-1))/(n*(n-1))
                else:
                    mu[(c1,c2)]=2*m*(c[c1]*c[c2])/(n*(n-1))
        pi3=self.pi3()
        var=dict()
        for c1 in list(c.keys()):
            for c2 in list(c.keys()):
                if c1==c2:
                    alpha=2*((self.fall(c[c1],3)/self.fall(n,3))-(self.fall(c[c1],4)/self.fall(n,4)))
                    beta=2*(self.fall(c[c1],4)/self.fall(n,4))
                    var[(c1,c2)]=mu[(c1,c2)]*(1-mu[(c1,c2)])+alpha*pi3+beta*comb(m,2)
                else:
                    alpha=2*(((c[c1]*self.fall(c[c2],2)+self.fall(c[c1],2)*c[c2])/self.fall(n,3))-(4*(self.fall(c[c1],2)*self.fall(c[c2],2)/self.fall(n,4))))
                    beta=8*(self.fall(c[c1],2)*self.fall(c[c2],2)/self.fall(n,4))
                    var[(c1,c2)]=mu[(c1,c2)]*(1-mu[(c1,c2)])+alpha*pi3+beta*comb(m,2)
        return var
    def rapporto_var_RC(self,ripetizioni=1000):#RAPPORTO VARIANZA ESATTA RC VS VARIANZA MONTECARLO
        from math import sqrt
        var_monte=self.var_RC_montecarlo(ripetizioni)
        var_ex=self.var_RC_esatta()
        rap=dict()
        for a in var_monte:
            rap[a]=var_ex[a]/var_monte[a]
        media=sum(list(rap.values()))/len(list(rap.values()))
        return rap
    def media_rap_var_RC(self,ripetizioni=1000):#MEDIA RAPPORTI VARIANZE RC
        rap=self.rapporto_var_RC(ripetizioni)
        media=sum(list(rap.values()))/len(list(rap.values()))
        return media
    def err_rel_var_RC(self,ripetizioni=1000):#ERRORI RELATIVI VARIANZE RC
        err=dict()
        var=self.var_RC_esatta()
        monte=self.var_RC_montecarlo(ripetizioni)
        for a in monte:
            err[a]=abs(var[a]-monte[a])/monte[a]
        return err,sum(err.values())/len(err.values())
    def plot_err_var_RC(self):#MEDIE ERRORI RELATIVI VARIANZE PER SIMULAZIONI RC
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [self.err_rel_var_RC(r)[1] for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2)
        plt.title("Var Teorica vs Var Montecarlo")
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Errore relativo medio varianza")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def scatter_RCvsCM(self,ripetizioni=1000): #SCATTER VARIANZE MONTECARLO RC VS VARIANZE MONTECARLO CM
        import matplotlib.pyplot as plt
        CM=self.var_CM_montecarlo(ripetizioni)
        RC=self.var_RC_montecarlo(ripetizioni)
        col = list(CM.keys())
        x = ([CM[c] for c in col])
        y = ([RC[c] for c in col])
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        plt.figure(figsize=(7,6))
        for c in col:
            plt.scatter(CM[c], RC[c],
            color="steelblue",
            s=80,
            edgecolor="black",
            alpha=0.8)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.xlabel("var CM")
        plt.ylabel("var RC")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

                    
    def indici_RC(self): #INDICI DI OMOFILIA GLOBALI E PARZIALI RC
        from math import sqrt
        mu=self.V_A_RC()
        var=self.var_RC_esatta()
        indici=dict()
        w=dict()
        m=len(self.lista_archi())
        n=len(self.lista_nodi())
        c=self.freq_col()
        s=len(list(c.keys()))
        archi_omo=self.omo_by_col()
        for c1 in list(archi_omo.keys()):
            w[(c1,c1)]=archi_omo[c1]/mu[(c1,c1)]
        indici['w intra']=w
        n=dict()
        archi_ete=self.ete_by_col()
        for coppia in list(archi_ete.keys()):
            n[coppia]=archi_ete[coppia]/mu[coppia]
        indici['n inter']=n
        z_score=dict()
        for c1 in list(c.keys()):
            for c2 in list(c.keys()):
                if c1==c2:
                    z_score[(c1,c2)]=(archi_omo[c1]-mu[(c1,c2)])/sqrt(var[(c1,c2)])
                else:
                    if c1>c2:
                        None
                    else:
                     z_score[(c1,c2)]=(archi_ete[(c1,c2)]-mu[(c1,c2)])/sqrt(var[(c1,c2)])
        indici['z scores']=z_score
        r=sum(list(archi_omo.values()))/m
        indici['r']=r
        Z=sum([z_score[c1,c2]  for c1 in list(c.keys())for c2 in list(c.keys()) if c1==c2 ])/s
        if Z>=0:
            segno=1
        else:
            segno=-1
        a=segno*(Z**2/(Z**2+s))
        indici['a']=a
        E=sum([z_score[c1,c2]**2  for c1 in list(c.keys())for c2 in list(c.keys()) if c1==c2 ])/s
        indici['E']=E
        Var_glo=dict()
        for c1 in list(c.keys()):
            for c2 in list(c.keys()):
                if c1==c2:
                    Var_glo[(c1,c2)]=(archi_omo[c1]-mu[(c1,c2)])**2/(var[(c1,c2)])
                else:
                    if c1>c2:
                        None
                    else:
                        Var_glo[(c1,c2)]=(archi_ete[(c1,c2)]-mu[(c1,c2)])**2/var[(c1,c2)]
        D=sum(list(Var_glo.values()))
        indici['D']=D
        
        
        return indici
    def indici_CM(self): #INDICI OMOFILIA CM
        from math import sqrt
        mu=self.V_A_CM()
        m=len(self.lista_archi())
        
        indici=dict()
        n=len(self.lista_nodi())
        c=self.freq_col()
        s=len(list(c.keys()))
        archi_omo=self.omo_by_col()

        gradi=self.dict_gradi()
        d={}
        for col in list(c.keys()):
            d[col]=0
        for g in list(gradi.keys()):
            d[self._col[g]]+=gradi[g]
        q=0
        for c1 in list(archi_omo.keys()):
            q+=archi_omo[c1]/m-(d[c1]/(2*m))**2
        indici['q']=q
        return indici
        
        
        
    
        
    
                
        
            
        
        
        
        
        
    def __str__(self):
        f=''
        return f
    
        

         
    
#GRAFO TEST   
    
g = {0: [33, 1, 3, 11, 25, 28],1: [0, 38, 7, 22, 27, 31],2: [],
    3: [0, 34, 35, 17, 20, 22, 23, 27, 30, 31],4: [7, 16, 17, 29, 30],
    5: [32, 34, 19, 37],6: [33, 21],7: [1, 4, 13, 15, 30],8: [33, 34, 12, 16, 25],
    9: [16, 34, 26],10: [35, 26, 18, 20],11: [0, 35, 12, 13, 18],12: [36, 8, 11, 17, 19, 23],
    13: [35, 7, 11, 14, 21],14: [34, 39, 13, 31],15: [24, 33, 34, 7],16: [4, 36, 8, 9, 18, 25, 26],
    17: [34, 3, 4, 12, 21, 29],18: [16, 10, 11],19: [5, 39, 12, 20, 22, 23, 26, 27],
    20: [32, 3, 10, 19, 23, 30],21: [32, 34, 37, 6, 13, 17, 29],22: [19, 1, 3, 28],
    23: [3, 36, 37, 12, 19, 20],24: [15],25: [0, 37, 8, 16, 31],26: [33, 35, 9, 10, 16, 19],
    27: [19, 1, 3],28: [32, 33, 0, 22, 29],29: [17, 21, 4, 28],30: [35, 3, 4, 38, 7, 20],
    31: [1, 3, 25, 14],32: [21, 28, 20, 5],33: [0, 36, 6, 8, 15, 26, 28],
    34: [3, 35, 5, 8, 9, 14, 15, 17, 21],35: [34, 3, 38, 10, 11, 13, 26, 30],
    36: [33, 37, 39, 12, 16, 23],37: [36, 5, 21, 23, 25],38: [1, 35, 30],39: [19, 36, 14]
}

gc = {
    0: 'blu',      1: 'rosso',   2: 'rosso',   3: 'giallo',  4: 'viola',
    5: 'blu',      6: 'rosso',   7: 'giallo',  8: 'rosso',   9: 'giallo',
    10: 'rosso',   11: 'viola',  12: 'rosso',  13: 'blu',    14: 'viola',
    15: 'verde',   16: 'verde',  17: 'verde',  18: 'verde',  19: 'viola',
    20: 'viola',   21: 'giallo', 22: 'giallo', 23: 'rosso',  24: 'viola',
    25: 'verde',   26: 'rosso',  27: 'rosso',  28: 'verde',  29: 'verde',
    30: 'verde',   31: 'rosso',  32: 'verde',  33: 'blu',    34: 'viola',
    35: 'blu',     36: 'giallo', 37: 'verde',  38: 'giallo', 39: 'rosso'
}




#DAL DIZIONARIO ALLA CLASSE

def grafo_da_diz(diz_archi,diz_col):
    gca=list(diz_col.keys())
    gcb=list(diz_col.values())
    ga=list(diz_archi.keys())
    gb=list(diz_archi.values())
    grafo = grafo_r()
    for i in range(len(gca)):
        grafo.add_nodo(gca[i],gcb[i])
    for nodonuovo in ga:
        for nodonuovo2 in gb[ga.index(nodonuovo)]:
            grafo.add_arco(nodonuovo,nodonuovo2)
    return(grafo)
if True == False:
    a=grafo_da_diz(g,gc)
    print(a.indici_CM())

       
#DALLA LISTA ARCHI AL GRAFO    
def grafo_da_lista(lista_archi,diz_col):
    gca=list(diz_col.keys())
    gcb=list(diz_col.values())
    grafo = grafo_r()
    for i in range(len(gca)):
        grafo.add_nodo(gca[i],gcb[i])
    for arco in lista_archi:
        grafo.add_arco(arco[0],arco[1])
    return(grafo)
#test
if True == False:
    import pandas as pd
    import random
    a=grafo_da_diz(g,gc)
    url = "https://raw.githubusercontent.com/MatteoBonimelli/RANDOM-NULL-GRAPHS/main/PAGES.csv"
    pages= pd.read_csv(url)
    url1="https://raw.githubusercontent.com/MatteoBonimelli/RANDOM-NULL-GRAPHS/main/LIKE.csv"
    like=pd.read_csv(url1)
    likes=like.sample(100)
    class_dict = dict(zip(pages['id'], pages['page_type']))
    list_like=[[int(likes['id_1'][i]),int(likes['id_2'][i])] for i in likes.index.tolist()]
    genera=grafo_da_lista(list_like,{})
    pages_chosen=dict()
    for i in genera.lista_nodi():
        pages_chosen[i]=class_dict[i]
    facebook=grafo_da_lista(list_like,pages_chosen)
    print(facebook.indici_RC())
    

 
#LETTURA LISTA DA TESTO
def leggi_lista(percorso):
    import ast
    with open(percorso, "r", encoding="utf-8") as testo:
        contenuto = testo.read()
    lista = ast.literal_eval(contenuto)    
    if not isinstance(lista, list):
        raise ValueError("Lista non presente o non letta")
    return lista
       
#LETTURA DIZIONARIO DA TESTO
def leggi_dict(percorso):
    import ast 
    with open(percorso, "r", encoding="utf-8") as testo:
        contenuto = testo.read()
    diz = ast.literal_eval(contenuto)
    if not isinstance(diz, dict):
        raise ValueError("Dizionario non presente o non letto")
    return diz


