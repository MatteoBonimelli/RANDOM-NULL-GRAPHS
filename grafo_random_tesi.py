class grafo_r:
    def __init__(self):
        self._nodi=dict()
        self._col=dict()
    
    def add_nodo(self,nodo,col=""):
        if nodo not in self._nodi:
            self._nodi[nodo]=[]
            self._col[nodo]=col
    def add_arco(self,nodo1,nodo2):
        if nodo1 not in self._nodi:
            self.add_nodo(nodo1)
        if nodo2 not in self._nodi:
            self.add_nodo(nodo2)
        if nodo2 not in self._nodi[nodo1]:
            self._nodi[nodo1].append(nodo2)
        if nodo1 not in self._nodi[nodo2]:
            self._nodi[nodo2].append(nodo1)
    def nodo_exist(self,nodo):
        return nodo in self._nodi
    def arco_exist(self,nodo1,nodo2):
        return (nodo1 in self._nodi[nodo2] and nodo1 in self._nodi[nodo2])
    def numero_nodi(self):
        return len(self._nodi)
    def numero_archi(self):
        a=0
        for n in self._nodi:
            a+=len(self._nodi[n])
        return a//2
    def lista_nodi(self):
        return list(self._nodi.keys())
    def lista_archi(self):
        archi=[]
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi:
                    archi.append([n,a])
        return archi
    def lista_archi_col(self):
        archi=[]
        numero=[]
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in numero:
                    numero.append([n,a])
                    archi.append([self._col[n],self._col[a]])
        return archi
    def lista_gradi(self):
        gradi=[]
        for n in self._nodi:
            gradi.append(len(self._nodi[n]))
        return gradi
    def dict_gradi(self):
        gradi={}
        for n in self._nodi:
            gradi[n]=len(self._nodi[n])
        return gradi
    def add_col(self,n,col):
        self._col[n]=col
    def freq_col(self):
        from collections import Counter
        return Counter(list(self._col.values()))
    def omo(self):
        archi=[]
        archi_omo=0
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi:
                    archi.append([n,a])
                    if self._col[n]==self._col[a]:
                        archi_omo+=1
        return archi_omo
    
            
    def archi_omo(self):
        archi=[]
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi: 
                    if self._col[n]==self._col[a]:
                        archi.append([n,a])
                        
        return archi
    def omo_by_col(self):
        omofili=dict()
        colori = sorted(list(set(self._col.values())))
        archi_col=self.lista_archi_col()
        for c in colori:
            count=0
            for a in range(len(archi_col)):
                if archi_col[a][0]==c and archi_col[a][1]==c:
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
                    if self._col[n]!=self._col[a]:
                        archi_ete+=1
        return archi_ete
    def archi_ete(self):
        archi=[]
        for n in self._nodi:
            for a in self._nodi[n]:
                if [a,n] not in archi: 
                    if self._col[n]!=self._col[a]:
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
                    if (c2,c1) not in eterofili.keys():
                        eterofili[(c1,c2)]=count
        return eterofili
        
    def random_coloring(self,ripetizioni=1):
        import random
        diz_rip=dict()
        nodi=list(self._col.keys())
        colori=list(self._col.values())
        for rip in range(ripetizioni):
            new_col=dict()
            colori_mix=random.sample(colori,k=len(colori))
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
        mezzi_archi_r=random.sample(mezzi_archi,k=len(mezzi_archi))
        archi=[]
        for i in range (0,len(mezzi_archi_r),2):
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
            raise ValueError('Somma gradi dispari')
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
        prova=self.random_coloring()
        return prova
    def set_CM(self):
        test=self.configuration_sw
        return test
    def media_RC(self):
        m=len(self.lista_archi())
        n=len(self.lista_nodi())
        c=self.freq_col()
        mu={}
        for c1 in list(c.keys()):
            for c2 in list(c.keys()):
                if c1==c2:
                    mu[(c1,c2)]=m*(c[c1]*(c[c1]-1))/(n*(n-1))
                elif c1<c2:
                    mu[(c1,c2)]=2*m*(c[c1]*c[c2])/(n*(n-1))
        return mu
    def media_CM(self):
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
            
    
        
    def mu_RC_montecarlo(self,ripetizioni=1000,varianza=False):
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
        val_att=self.media_RC()
        media=self.mu_RC_montecarlo()
        diff=dict()
        for a in val_att:
                diff[a]=val_att[a]-media[a]
        return diff
    def diff_mu_perc_RC(self,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.media_RC()
        media=self.mu_RC_montecarlo(ripetizioni)
        diff=dict()
        for a in val_att:
                diff[a]=(val_att[a]-media[a])/val_att[a]
        return diff,sum(diff.values())/len(diff.values())
    
    def media_diff_mu_RC(self):
        diff=self.diff_mu_RC()
        media=sum(list(diff.values()))/len(list(diff.values()))
        return media
    def errore_mu_RC(self,relativo=False,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.media_RC()
        media=self.mu_RC_montecarlo(ripetizioni=ripetizioni)
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
        err=self.errore_mu_RC(True,ripetizioni)[0]
        media=sum(err.values())/len(err.values())
        return media
    def plot_err_rip_RC(self):
        import matplotlib.pyplot as plt
        rip = [50, 100, 200, 500, 1000,2000,5000,10000,20000,50000]

        medie = [self.media_err_RC(r) for r in rip]
        plt.figure(figsize=(7,5))
        plt.plot(rip, medie, marker='o', linewidth=2)
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
        plt.plot(rip, medie, marker='o', linewidth=2)
        plt.xscale('log')   
        plt.xlabel("Simulazioni")
        plt.ylabel("Scarto relativo medio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    def var_RC_montecarlo(self,ripetizioni=1000):
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
    
    
    
    
    def mu_CM_montecarlo(self,ripetizioni=1000,varianza=False):
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
    
    
    def diff_mu_CM(self,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.media_CM()
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
    def diff_mu_perc_CM(self,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.media_CM()
        media=self.mu_CM_montecarlo(ripetizioni)
        diff=dict()
        for a in val_att:
                diff[a]=(val_att[a]-media[a])/val_att[a]
                
        return diff,sum(diff.values())/len(diff.values())
    def errore_mu_CM(self,relativo=False,ripetizioni=1000):
        c=self.freq_col()
        col=list(c.keys())
        val_att=self.media_CM()
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
    def media_err_CM(self,ripetizioni=1000):
        err=self.errore_mu_CM(True,ripetizioni)[0]
        media=sum(err.values())/len(err.values())
        return media
        
    
    def media_diff_mu_CM(self,ripetizioni=1000):
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
    def z_score_CM(self,omo=False,ete=False):
        from math import sqrt
        val_att=self.media_CM()
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
    def scatter_val_att_vs_montecarlo_CM(self):
        import matplotlib.pyplot as plt
        val_att=self.media_CM()
        media=self.mu_CM_montecarlo()
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
    def scatter_val_att_vs_montecarlo_RC(self):
        import matplotlib.pyplot as plt
        val_att=self.media_RC()
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

   
            

        
    def dist_media_diff_mu_CM(self,ripetizioni=10):
        import numpy as np
        rip_medie=[]
        rip_omo=[]
        rip_ete=[]
        rip_abs=[]
        for i in range(ripetizioni):
            ripetizione=self.media_diff_mu_CM()
            rip_medie.append(ripetizione[0])
            rip_omo.append(ripetizione[1])
            rip_ete.append(ripetizione[2])
            rip_abs.append(ripetizione[3])
        medie=np.array(rip_medie)
        omo=np.array(rip_omo)
        ete=np.array(rip_ete)
        abso=np.array(rip_abs)
        ris={'media medie':float(np.mean(medie)),'varianza medie':float(np.var(medie,ddof=1)),
             'media medie omofile':float(np.mean(omo)),'varianza medie omofile':float(np.var(omo,ddof=1)),
             'media medie eterofile':float(np.mean(ete)),'varianza medie eterofile':float(np.var(ete,ddof=1)),
             'media medie valori assoluti':float(np.mean(abso)),'varianza medie valori assoluti':float(np.var(abso,ddof=1))}
        

        return ris
    def IC_CM(self):
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
        v_a=self.media_CM()
        in_int=dict()
        for co in v_a:
            in_int[co]=v_a[co]>IC[co][0] and v_a[co]<IC[co][1]
            
        return IC,in_int,sum(in_int.values())/len((in_int.values()))*100
    def plot_err_rip_CM(self):
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
    
    def plot_diff_rip_CM(self):
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

    
        
        
        
            
    def var_CM_montecarlo(self,ripetizioni=1000):
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
    def var_CM(self):
        mu=self.media_CM()
        m=len(self.lista_archi())
        var=dict()
        for a in mu:
            var[a]=mu[a]*(1-mu[a]/(2*m))
        return var
    def rap_var_CM(self,ripetizioni=1000):
        rap=dict()
        var=self.var_CM()
        monte=self.var_CM_montecarlo(ripetizioni)
        for a in var:
            rap[a]=var[a]/monte[a]
        return rap,sum(rap.values())/len(rap.values())
    def err_rel_var_CM(self,ripetizioni=1000):
        err=dict()
        var=self.var_CM()
        monte=self.var_CM_montecarlo(ripetizioni)
        for a in var:
            err[a]=abs(var[a]-monte[a])/monte[a]
        return err,sum(err.values())/len(err.values())
    def plot_err_var_CM(self):
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
    def plot_rap_var_CM(self):
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
        
    def fall(self,x,k):
        ris=1
        for i in range(k):
            ris*=(x-i)
        return ris
    def pi3(self):
        from math import comb
        pi3=sum(comb(k,2) for k in self.lista_gradi() if k>=2)
        return pi3
    def var_RC_esatta(self):
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
    def rapporto_var_RC(self,ripetizioni=1000):
        from math import sqrt
        var_monte=self.var_RC_montecarlo(ripetizioni)
        var_ex=self.var_RC_esatta()
        rap=dict()
        for a in var_monte:
            rap[a]=var_ex[a]/var_monte[a]
        media=sum(list(rap.values()))/len(list(rap.values()))
        return rap
    def media_rap_var_RC(self,ripetizioni=1000):
        rap=self.rapporto_var_RC(ripetizioni)
        media=sum(list(rap.values()))/len(list(rap.values()))
        return media
    def err_rel_var_RC(self,ripetizioni=1000):
        err=dict()
        var=self.var_RC_esatta()
        monte=self.var_RC_montecarlo(ripetizioni)
        for a in monte:
            err[a]=abs(var[a]-monte[a])/monte[a]
        return err,sum(err.values())/len(err.values())
    def plot_err_var_RC(self):
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
    def scatter_RCvsCM(self):
        import matplotlib.pyplot as plt
        CM=self.var_CM_montecarlo()
        RC=self.var_RC_montecarlo()
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

                    
    def indici_RC(self):
        from math import sqrt
        mu=self.media_RC()
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
    def indici_CM(self):
        from math import sqrt
        mu=self.media_CM()
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
        f='\n'
        #f+=f'Z SCORES CM {self.z_score_CM()}\n'
        #f+=f'DIFFERENZA VALORE ATTESO MENO MEDIE MONTECARLO CM {self.diff_mu_CM()[0]}\n'
        f+=f' {self.scatter_RCvsCM()}\n'
        f+='\n'
        #f+=f'MEDIA DIFFERENZE CM {self.media_diff_mu_CM()[0]} MEDIA ARCHI OMOFILI {self.media_diff_mu_CM()[1]}  MEDIA ARCHI ETEROFILI {self.media_diff_mu_CM()[2]} MEDIA VALORI ASSOLUTI {self.media_diff_mu_CM()[3]} \n'
        f+='\n'
        #f+=f'DISTRIBUZIONE MEDIA DIFFERENZE CM {self.dist_media_diff_mu_CM()}'
        f+='\n'
        #f+=f'DIFFERENZA VALORE ATTESO MENO MEDIE MONTECARLO RC {self.diff_mu_RC()}\n'
        f+='\n'
        #f+=f'MEDIA DIFFERENZE RC {self.media_diff_mu_RC()}\n'
        f+='\n'
        #f+=f'VARIANZA CM MONTECARLO {self.var_CM_montecarlo()}\n'
        f+='\n'
        #f+=f'VARIANZA RC ESATTA {self.var_RC_esatta()}\n'
        f+='\n'
        #f+=f'VARIANZA RC APPROSSIMATA {self.var_RC_approx()}\n'
        f+='\n'
        #f+=f'VARIANZA RC MONTECARLO {self.var_RC_montecarlo()}\n'
        f+='\n'
        #f+=f'RAPPORTO VARIANZA ESATTA/VARIANZA MONTECARLO RC {self.rapporto_var_RC2()}\n'
        f+='\n'
        #f+=f'MEDIA RAPPORTO VARIANZA ESATTA/VARIANZA MONTECARLO RC {self.media_rap_var_RC2()}\n'
        f+='\n'
        #f+=f'RAPPORTO VARIANZA APPROSSIMATA/VARIANZA MONTECARLO RC {self.rapporto_var_RC()}\n'
        f+='\n'
        #f+=f'MEDIA RAPPORTO VARIANZA APPROSSIMATA/VARIANZA MONTECARLO RC {self.media_rap_var_RC()}\n'
        f+='\n'
        #f+=f'indici Random Coloring {self.indici_RC()}\n'
        f+='\n'
        #f+=f'indici Configuration Model {self.indici_CM()}\n'
        
        
        
        
        
        
        return f
    
        

         
    
    
    
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

g1 = {
    0: [2, 7],
    1: [18, 83],
    2: [0, 22],
    3: [9, 35],
    4: [23, 42],
    5: [9, 65],
    6: [19, 85],
    7: [0, 34],
    8: [16, 73],
    9: [3, 5],
    10: [40, 79],
    11: [27, 91],
    12: [33, 55],
    13: [19, 90],
    14: [48, 82],
    15: [17, 95],
    16: [8, 84],
    17: [15, 64],
    18: [1, 47],
    19: [6, 13],
    20: [46, 74],
    21: [37, 70],
    22: [2, 80],
    23: [4, 49],
    24: [58, 62],
    25: [53, 85],
    26: [31, 90],
    27: [11, 75],
    28: [56, 91],
    29: [41, 98],
    30: [36, 63],
    31: [26, 77],
    32: [38, 93],
    33: [12, 59],
    34: [7, 92],
    35: [3, 39],
    36: [30, 66],
    37: [21, 68],
    38: [32, 60],
    39: [35, 89],
    40: [10, 81],
    41: [29, 71],
    42: [4, 44],
    43: [78, 97],
    44: [42, 61],
    45: [49, 95],
    46: [20, 87],
    47: [18, 50],
    48: [14, 99],
    49: [23, 45],
    50: [47, 58],
    51: [55, 83],
    52: [67, 80],
    53: [25, 79],
    54: [70, 96],
    55: [12, 51],
    56: [28, 81],
    57: [64, 98],
    58: [24, 50],
    59: [33, 86],
    60: [38, 73],
    61: [44, 88],
    62: [24, 84],
    63: [30, 99],
    64: [17, 57],
    65: [5, 74],
    66: [36, 82],
    67: [52, 87],
    68: [37, 94],
    69: [72, 93],
    70: [21, 54],
    71: [41, 97],
    72: [69, 75],
    73: [8, 60],
    74: [20, 65],
    75: [27, 72],
    76: [86, 96],
    77: [31, 99],
    78: [43, 94],
    79: [10, 53],
    80: [22, 52],
    81: [40, 56],
    82: [14, 66],
    83: [1, 51],
    84: [16, 62],
    85: [6, 25],
    86: [59, 76],
    87: [46, 67],
    88: [61, 92],
    89: [39, 95],
    90: [13, 26],
    91: [11, 28],
    92: [34, 88],
    93: [32, 69],
    94: [68, 78],
    95: [15, 45, 89],
    96: [54, 76],
    97: [43, 71],
    98: [29, 57],
    99: [48, 63, 77]
}

gc1 = {
    0: 'rosso', 1: 'verde', 2: 'blu', 3: 'giallo', 4: 'rosso',
    5: 'blu', 6: 'viola', 7: 'verde', 8: 'giallo', 9: 'rosso',
    10: 'blu', 11: 'viola', 12: 'verde', 13: 'rosso', 14: 'giallo',
    15: 'rosso', 16: 'blu', 17: 'verde', 18: 'giallo', 19: 'viola',
    20: 'rosso', 21: 'verde', 22: 'blu', 23: 'giallo', 24: 'rosso',
    25: 'blu', 26: 'viola', 27: 'verde', 28: 'giallo', 29: 'rosso',
    30: 'blu', 31: 'verde', 32: 'giallo', 33: 'rosso', 34: 'blu',
    35: 'viola', 36: 'verde', 37: 'giallo', 38: 'rosso', 39: 'blu',
    40: 'verde', 41: 'giallo', 42: 'rosso', 43: 'viola', 44: 'verde',
    45: 'giallo', 46: 'rosso', 47: 'blu', 48: 'verde', 49: 'giallo',
    50: 'rosso', 51: 'blu', 52: 'viola', 53: 'verde', 54: 'giallo',
    55: 'rosso', 56: 'blu', 57: 'verde', 58: 'giallo', 59: 'rosso',
    60: 'blu', 61: 'verde', 62: 'giallo', 63: 'rosso', 64: 'blu',
    65: 'viola', 66: 'verde', 67: 'giallo', 68: 'rosso', 69: 'blu',
    70: 'verde', 71: 'giallo', 72: 'rosso', 73: 'blu', 74: 'verde',
    75: 'giallo', 76: 'rosso', 77: 'blu', 78: 'verde', 79: 'giallo',
    80: 'rosso', 81: 'blu', 82: 'verde', 83: 'giallo', 84: 'rosso',
    85: 'blu', 86: 'verde', 87: 'giallo', 88: 'rosso', 89: 'blu',
    90: 'verde', 91: 'giallo', 92: 'rosso', 93: 'blu', 94: 'verde',
    95: 'giallo', 96: 'rosso', 97: 'blu', 98: 'verde', 99: 'giallo'
}






def grafo_crea_diz(diz_archi,diz_col):
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
    print(grafo)
    return(grafo)
if __name__ == "__main__":
    a=grafo_crea_diz(g,gc)
    
    
def grafo_crea_diz_lista(lista_archi,diz_col):
    gca=list(diz_col.keys())
    gcb=list(diz_col.values())
    grafo = grafo_r()
    for i in range(len(gca)):
        grafo.add_nodo(gca[i],gcb[i])
    for arco in lista_archi:
        grafo.add_arco(arco[0],arco[1])
    print(grafo)
    return(grafo)


grafo_lettere=dict()
colori_lettere=dict()
for i in list(g.keys()):
    grafo_lettere[f'N{i}']=[]
    colori_lettere[f'N{i}']=gc[i]

  
    
for i in list(g.keys()):    
    for a in range(len(g)):
        for n in g[a]:
            grafo_lettere[f"N{a}"].append(f"N{n}")
            
        


    
    
    
   

def leggi_lista(percorso):
    import ast
    with open(percorso, "r", encoding="utf-8") as testo:
        contenuto = testo.read()
    lista = ast.literal_eval(contenuto)    
    if not isinstance(lista, list):
        raise ValueError("Lista non presente o non letta")
    return lista
       

def leggi_dict(percorso):
    import ast 
    with open(percorso, "r", encoding="utf-8") as testo:
        contenuto = testo.read()
    diz = ast.literal_eval(contenuto)
    if not isinstance(diz, dict):
        raise ValueError("Dizionario non presente o non letto")
    return diz


