import numpy as np
import pandas as pd
import sys
from htmd import *
class Contactome(pd.DataFrame):
    "pandas.DataFrame child class with some specific methods.The methods works with an specific multi-index characteristic of this class."
    @property
    def _constructor(self):
        return Contactome

    def get_chn_int(self):
        """Return a pd.DataFrame object with the entries of the contactome with interactions with other chains.

        Returns
        -------
        pandas.DataFrame : Pandas object
        """
        if len(self.index.levels[0]) < 2:
            sys.stderr.write("**WARNING: This protein only has one chain:'%s'\n"%self.index.levels[0][0])
            sys.exit()
        results = set([])
        for chain in np.unique(self.index.get_level_values('Chain')):
            data = self.iloc[self.index.get_level_values('Chain') == chain]
            for index, column in data.loc[~data['chain'].isin([chain,''])].groupby(level=([0,1,3,4])):
                if (column['resid'][0],index[1]) not in results:
                    results.add((index[0],index[1],column['chain'][0],column['resid'][0]))
        res = np.array(list(results))
        return pd.DataFrame(res[res[:,0].argsort()], columns=['Chain1','Resid1','Chain2','Resid2'])

    def narrow(self,cutoff=0.5):
        """Return a new Contactome object with the first hit (n=1) and all the following hits with a difference between bond-distance equal o lower than a given cutoff.

        Parameters
        ----------
        cutoff : float
            Set the maximum difference between the bond-distance of the first hit and the others to be included in the new Contactome object

        Returns
        -------
        ctctome : pandas.DataFrame.Contactome object
        """
        my_dict = {}
        for n, data in self.groupby(level=([0,1,2,3])):
            if len(data) == 1:
                my_dict[data.iloc[0].name] = np.asarray(list(data.iloc[0]))
            elif len(data) > 1:
                bonds = data['bond']
                if abs(bonds[0]-bonds[1]) > cutoff:
                    my_dict[data.iloc[0].name] = np.asarray(list(data.iloc[0]))
                else:
                    for i in range(0,len(bonds)):
                        if abs(bonds[0]-bonds[i]) < cutoff:
                            my_dict[data.iloc[i].name] = np.asarray(list(data.iloc[i]))
                        else:
                            break
        ctctome = Contactome(my_dict)
        ctctome.rename(index={0: 'chain',1:'resid',2:'resname',3:'name',4:'bond'}, inplace=True)
        ctctome = ctctome.transpose()
        ctctome.index.names = ['Chain','Resid','Resname','Name','n']
        ctctome.loc[ctctome['resid'] != '','resid'] = ctctome.loc[ctctome['resid'] != '','resid'].astype(int)
        ctctome.loc[ctctome['bond'] != '','bond'] = ctctome.loc[ctctome['bond'] != '','bond'].astype(float)
        return(ctctome)

    def check(self,pqr,cutoff=3.5):
        """Call the functions 'charge_test' and 'his_test' and return a pd.Dataframe object with a summary of the results for both functions.

        Parameters
        ----------
        pqr : str
            The variable name of the proteinPrepare result (pqr_object)
        cutoff : float
            Distance cutoff for the specific comprobations for the charge_test and his_test functions. Should be the same used for the contactome_maker function.

        Returns
        -------
        pandas.Dataframe : Pandas object
        """
        my_dict ={'Chain':[],'Resid':[],'Resname':[],'Problem':[],'Hit':[]}
        chrg_dict = Contactome.charge_test(self,pqr,cutoff)
        sys.stderr.write("\n")
        his_dict = Contactome.his_test(self,pqr,cutoff)
        for key,value in chrg_dict.items():
            if len(value) != 1:
                my_list = list(value[0])
                for i in range(1,len(value)):
                    my_list.append(value[i][3])
                    value = tuple(my_list)
            else:
                value = value[0]
            my_dict['Chain'].append(key[0])
            my_dict['Resid'].append(int(key[1]))
            my_dict['Resname'].append(key[2])
            if key[2] in ('GLU','ASP'):
                my_dict['Problem'].append('-')
            else:
                my_dict['Problem'].append('+')
            my_dict['Hit'].append(value)

        for key,value in his_dict.items():
            my_dict['Chain'].append(key[0])
            my_dict['Resid'].append(int(key[1]))
            my_dict['Resname'].append(key[2])
            my_dict['Problem'].append('-->')
            my_dict['Hit'].append(" ".join(value))

        return pd.DataFrame(my_dict)[['Chain','Resid','Resname','Problem','Hit']]

    def charge_test(contactome,pqr,cutoff=3.5):
        """Return a dictionary with the aminoacids from the Contactome object with a charge clash issue

        Parameters
        ----------
        contactome : pandas.DataFrame.Contactome object
            The variable name where the Contatome object is saved
        pqr : str
            The variable name of the proteinPrepare result (pqr_object)
        cutoff : float
            Distance cutoff for the specific comprobations. Should be the same used for the contactome_maker function.

        Returns
        -------
        charge_report :  dictionary
            Contains tuples with the Contactome aa (source) information as a key and tuples with the hit (target) information as a value
        """
        cations = ('CA','K','MG','NA','ZN','FE','MN','CS','CU','HG','HO3','LI','NH4','NI','RB','RU')
        anions = ('SO4','CL','PO4','BR','IOD','ACT','CO3','CYN','FLC','LCP','NO2','NO3','OH','OXL','PER','SCN','SO3')
        pos_atoms = ('NH1','NH2','NZ','ND1','ND2')
        neg_atoms = ('OD1','OD2','OE1','OE2')
        pos_aa = {
        'ARG':('NH1','NH2'),
        'LYS':('NZ',),
        'HIP':('ND1','NE2'),
        }
        neg_aa = {
        'ASP':('OD1','OD2'),
        'GLU':('OE1','OE2'),
        }
        charge_report = {}
        # NEGATIVE CHARGES TEST
        for neg,columns in contactome.loc(axis=0)[:,:,tuple(neg_aa.keys())].groupby(level=([0,1,2,3])): # iterar per una llista dels GLU i ASP del contactoma
            clash_resid = pqr.get("resid",sel="not resid "+str(neg[1])+" and ((resname "+" ".join(neg_aa.keys())+" and name "+" ".join(neg_atoms)+") or (resname "+" ".join(anions)+")) and within "+str(cutoff)+" of (chain "+neg[0]+" and resid "+str(neg[1])+" and name "+neg[3]+")")
            if len(clash_resid) != 0:
                clash_resname = pqr.get("resname",sel="not resid "+str(neg[1])+" and ((resname "+" ".join(neg_aa.keys())+" and name "+" ".join(neg_atoms)+") or (resname "+" ".join(anions)+")) and within "+str(cutoff)+" of (chain "+neg[0]+" and resid "+str(neg[1])+" and name "+neg[3]+")")
                clash_name = pqr.get("name",sel="not resid "+str(neg[1])+" and ((resname "+" ".join(neg_aa.keys())+" and name "+" ".join(neg_atoms)+") or (resname "+" ".join(anions)+")) and within "+str(cutoff)+" of (chain "+neg[0]+" and resid "+str(neg[1])+" and name "+neg[3]+")")
                clash_chain = pqr.get("chain",sel="not resid "+str(neg[1])+" and ((resname "+" ".join(neg_aa.keys())+" and name "+" ".join(neg_atoms)+") or (resname "+" ".join(anions)+")) and within "+str(cutoff)+" of (chain "+neg[0]+" and resid "+str(neg[1])+" and name "+neg[3]+")")
                hits = list(zip(clash_chain,clash_resid,clash_resname,clash_name))
                for i in hits:
                    if i in charge_report.keys():
                        hits.remove(i)
                if len(hits) > 0:
                    charge_report[neg] = hits
        # POSITIVE CHARGES TEST
        for pos,columns in contactome.loc(axis=0)[:,:,tuple(pos_aa.keys())].groupby(level=([0,1,2,3])): # iterar per una llista dels LYS, ARG i HIP del contactoma
            clash_resid = pqr.get("resid",sel="not resid "+str(pos[1])+" and ((resname "+" ".join(pos_aa.keys())+" and name "+" ".join(pos_atoms)+") or (resname "+" ".join(cations)+")) and within "+str(cutoff)+" of (chain "+pos[0]+" and resid "+str(pos[1])+" and name "+pos[3]+")")
            if len(clash_resid) != 0:
                clash_resname = pqr.get("resname",sel="not resid "+str(pos[1])+" and ((resname "+" ".join(pos_aa.keys())+" and name "+" ".join(pos_atoms)+") or (resname "+" ".join(cations)+")) and within "+str(cutoff)+" of (chain "+pos[0]+" and resid "+str(pos[1])+" and name "+pos[3]+")")
                clash_name = pqr.get("name",sel="not resid "+str(pos[1])+" and ((resname "+" ".join(pos_aa.keys())+" and name "+" ".join(pos_atoms)+") or (resname "+" ".join(cations)+")) and within "+str(cutoff)+" of (chain "+pos[0]+" and resid "+str(pos[1])+" and name "+pos[3]+")")
                clash_chain = pqr.get("chain",sel="not resid "+str(pos[1])+" and ((resname "+" ".join(pos_aa.keys())+" and name "+" ".join(pos_atoms)+") or (resname "+" ".join(cations)+")) and within "+str(cutoff)+" of (chain "+pos[0]+" and resid "+str(pos[1])+" and name "+pos[3]+")")
                hits = list(zip(clash_chain,clash_resid,clash_resname,clash_name))
                for i in hits:
                    if i in charge_report.keys():
                        hits.remove(i)
                if len(hits) > 0:
                    charge_report[pos] = hits
        # ANIONS AND CATIONS COMPROVATION
        for first,second in charge_report.items():
            if first[2] in " ".join(neg_aa.keys()):
                cat1 = pqr.get("resid", sel="resname "+" ".join(cations)+" and within "+str(cutoff)+" of (chain "+first[0]+" and resid "+str(first[1])+" and name "+first[3]+")")
                if len(cat1) != 0:
                    for target in second:
                        cat2 = pqr.get("resid", sel="resname "+" ".join(cations)+" and within "+str(cutoff)+" of (chain "+target[0]+" and resid "+str(target[1])+" and name "+target[3]+")")
                        if len(cat2) != 0:
                            charge_report[first].remove(target)
            if first[2] in " ".join(pos_aa.keys()):
                an1 = pqr.get("resid", sel="resname "+" ".join(anions)+" and within "+str(cutoff)+" of (chain "+first[0]+" and resid "+str(first[1])+" and name "+first[3]+")")
                if len(an1) != 0:
                    for target in second:
                        an2 = pqr.get("resid", sel="resname "+" ".join(anions)+" and within "+str(cutoff)+" of (chain "+target[0]+" and resid "+str(target[1])+" and name "+target[3]+")")
                        if len(an2) != 0:
                            charge_report[first].remove(target)
        # REMOVAL OF EMPTY KEYS IN DICTIONARY
        empty_keys = []
        for key in charge_report:
            if charge_report[key] == []:
                empty_keys.append(key)
        for i in empty_keys:
            charge_report.pop(i)
        # MESSAGE OF THE REPORT
        if len(charge_report) != 0:
            sys.stderr.write("\nCHARGE CLASHES:  (cutoff=%s)\n"%(cutoff))
        for first,second in charge_report.items():
            for i in second:
                sys.stderr.write("Charged residue %s is in direct H-bond orientation with a residue of the same charge: %s\n"%(first,i))
        return charge_report # OR return charge_warning IF WE WANT THE REPORT IN TEXT FORMAT
    def his_judge(evidences,tag='ND1'):
        """Return a protonation name for a histidine based on the bond distances of the donors and acceptors atoms from the histidine atoms ND1 and NE2
        Parameters
        ----------
        evidences : tuple/list of floats
            It must contain the bond distances with the following order: (ND1a,ND1d,NE2a,NE2d) ; where 'a' means treating the atom as an acceptor and 'd' as a donor. If not atom avaiable, put '0' or 'False'
        tag : str
            The pre-given protonation state of the histidine. It will be used if all options could be possible or if the information is not accurate enough

        Returns
        -------
        solution :  str
            String with the protonation name for the histidine based on its bond-distances.
        """
        if len(evidences) != 4:
            sys.stderr.write("Too many positions in the histidine array: %s (4 expected)"%len(evidences))
            sys.exit()

        solution = set([])
        candidates = np.flatnonzero(evidences)
        if len(candidates) == 1:
            suspects = ('HIE','HID','HID','HIE')
            solution.add(suspects[np.asscalar(candidates)])
        elif len(candidates) == 2:
            if abs(np.asscalar(np.diff(candidates))) == 2:
                suspects = ('HIE','HID','HID','HIE')
                if 0 in candidates:
                    [solution.add(suspects[np.asscalar(x)]) for x in np.flatnonzero(evidences== np.min(evidences[np.nonzero(evidences)]))]
                else:
                    solution.add('HIP')
            elif abs(np.asscalar(np.diff(candidates))) == 1:
                suspects = ('HIE','HID','HID','HIE')
                if any(x in candidates for x in [0, 3]):
                    [solution.add(suspects[np.asscalar(x)]) for x in np.flatnonzero(evidences == np.min(evidences[np.nonzero(evidences)]))]
                else:
                    solution.add('HID')
            else:
                solution.add('HIE')
        elif len(candidates) == 3:
            suspects = ('HIE','HID')
            suspects2 = ('HIE','HIP','HID','HIP')
            candidates2 = np.flatnonzero(evidences==0)
            if any(x in candidates2 for x in [1, 3]):
                solution.add(suspects[max(0,np.asscalar(candidates2)-2)])
            elif 0 in candidates2:
                [solution.add(suspects2[np.asscalar(x)]) for x in np.flatnonzero(evidences == np.min([evidences[2],evidences[3]]))]
            elif 2 in candidates2:
                [solution.add(suspects2[np.asscalar(x)]) for x in np.flatnonzero(evidences == np.min([evidences[0],evidences[1]]))]
        elif len(candidates) == 4:
            reduction = np.array([])
            reduction = np.append(reduction,[np.asscalar(x) for x in np.flatnonzero(evidences == np.min([evidences[0],evidences[1]]))])
            reduction = np.append(reduction,[np.asscalar(x) for x in np.flatnonzero(evidences == np.min([evidences[2],evidences[3]]))])
            if len(reduction) == 2:
                if abs(np.asscalar(np.diff(reduction))) == 1:
                    solution.add('HID')
                elif abs(np.asscalar(np.diff(reduction))) == 2:
                    if 3 in reduction:
                        solution.add('HIP')
                    else:
                        suspects = ('HIE','HID')
                        [solution.add(x) for x in suspects[max(0,np.asscalar(np.flatnonzero(evidences == np.min([evidences[0],evidences[2]])))-1)]]
                else:
                    solution.add('HIE')
            elif len(reduction) == 3:
                if 0 not in reduction:
                    [solution.add(x) for x in ('HID','HIP')]
                elif 1 not in reduction:
                    solution.add('HIE')
                elif 2 not in reduction:
                    [solution.add(x) for x in ('HIE','HID')]
                else:
                    solution.add('HID')
            else:
                solution.add(tag)
        else:
            solution.add(tag)
        return solution

    def his_test(contactome,pqr,cutoff=3.5):
        """Return a dictionary with the aminoacids from the Contactome object with a histidine protonation issue

        Parameters
        ----------
        contactome : pandas.DataFrame.Contactome object
            The variable name where the Contatome object is saved
        pqr : str
            The variable name of the proteinPrepare result (pqr_object)
        cutoff : float
            Distance cutoff for the specific comprobations. Should be the same used for the contactome_maker function.

        Returns
        -------
        his_dict :  dictionary
            Contains tuples with the Contactome aa (source) information as a key and tuples with the hit (target) information as a value
        """
        histidine =['HIP','HID','HIE']
        cations = ('CA','K','MG','NA','ZN','FE','MN','CS','CU','HG','HO3','LI','NH4','NI','RB','RU')
        anions = ('SO4','CL','PO4','BR','IOD','ACT','CO3','CYN','FLC','LCP','NO2','NO3','OH','OXL','PER','SCN','SO3')
        his_dict = {}
        for his in histidine:
            for chain in contactome.index.levels[0]:
                try:
                    filtered = contactome.loc(axis=0)[chain,:,his,:]
                except:
                    continue
                for hit,data2 in filtered.groupby(level=[0,1]):
                    array = []
                    for atom,data in data2.groupby(level=3):
                        if atom =='ND1d' or atom =='NE2d':
                            hits = []
                            reverse_output = ""
                            if data['resid'].iloc[0]== "":
                                array.append(False)
                            elif len(data['resid']) == 1:
                                hits.append((data['chain'].iloc[0],float(data['resid'].iloc[0]),data['resname'].iloc[0],data['name'].iloc[0]))
                                if hits[0] not in contactome:
                                    res = np.unique(pqr.get('resid',sel="not resid "+str(data['resid'].iloc[0])+" and ((name OH OG OG1 HE2 NE NH1 NH2 NZ ND1 NE2 ND2 and resname TYR SER THR TRP ARG AR0 LYS LYN HID HIE HIP GLN ASN) or resname CA K MG NA ZN FE MN CS CU HG HO3 LI NH4 NI RB RU or name H) and within "+str(cutoff)+" of (chain "+data['chain'].iloc[0]+" and name "+ data['name'].iloc[0]+ " and resid "+str(data['resid'].iloc[0])+")"))
                                    if len(res) == 1 and hit[1] in res:
                                        array.append(data['bond'].iloc[0])
                                    else:
                                        array.append(1000)
                                else:
                                    reverse_output = contactome.loc(axis=0)[hits[0]]
                                    if len(reverse_output) == 1 and reverse_output['resname'].iloc[0] == his:
                                        array.append(reverse_output['bond'].iloc[0])
                                    elif len(reverse_output) > 1:
                                        if reverse_output.iloc[0,2] == his:
                                            substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                            if substract >= 0.5:
                                                array.append(reverse_output['bond'][0].iloc[0])
                                            else:
                                                array.append(1000)
                                        elif reverse_output.iloc[0,2] != his:
                                            substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.loc[reverse_output['resname']==his]['bond'].iloc[0])))
                                            if substract <= 0.5:
                                                array.append(1000)
                                            else:
                                                array.append(5000)
                            else:
                                decision_array = []
                                for num, data3 in data.groupby(level=4):
                                    hits.append((data3['chain'].iloc[0],float(data3['resid'].iloc[0]),data3['resname'].iloc[0],data3['name'].iloc[0]))
                                for element in hits:
                                    if hits[0] not in contactome:
                                        res = np.unique(pqr.get('resid',sel="not resid "+str(data['resid'].iloc[0])+" and ((name OH OG OG1 HE2 NE NH1 NH2 NZ ND1 NE2 ND2 and resname TYR SER THR TRP ARG AR0 LYS LYN HID HIE HIP GLN ASN) or resname CA K MG NA ZN FE MN CS CU HG HO3 LI NH4 NI RB RU or name H) and within "+str(cutoff)+" of (chain "+data['chain'].iloc[0]+" and name "+ data['name'].iloc[0]+ " and resid "+str(data['resid'].iloc[0])+")"))
                                        if len(res) == 1 and hit[1] in res:
                                            decision_array.append(data['bond'].iloc[0])
                                        else:
                                            decision_array.append(1000)
                                    else:
                                        reverse_output=(contactome.loc(axis=0)[element])
                                        if len(reverse_output) == 1 and reverse_output.iloc[0,2]==hits:
                                            decision_array.append(reverse_output['bond'].iloc[0])
                                        elif len(reverse_output) > 1:
                                            if reverse_output['resname'].iloc[0] != his:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.loc[reverse_output['resname']==his]['bond'].iloc[0])))
                                                if substract <= 0.5:
                                                    decision_array.append(1000)
                                                else:
                                                    decision_array.append(5000)
                                            else:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                                if substract >= 0.5:
                                                    decision_array.append(reverse_output['bond'].iloc[0])
                                                else:
                                                    decision_array.append(1000)
                                array.append(min(decision_array))
                        if atom =='ND1a' or atom =='NE2a':
                            hits = []
                            bonds=[]
                            reverse_output=''
                            if data['resid'].iloc[0] == "":
                                array.append(False)
                            elif len(data['resid']) == 1:
                                hits.append((data['chain'].iloc[0],float(data['resid'].iloc[0]),data['resname'].iloc[0],data['name'].iloc[0]))
                                bonds.append(data['bond'].iloc[0])
                                if hits[0] not in contactome:
                                    res = np.unique(pqr.get('resid',sel="not resid "+str(hits[0][1])+" and ((name OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH and resname ASN ASP ASH GLN GLU GLH GLU HIP HID HIE SER THR TYR) or resname SO4 CL PO4 BR IOD ACT CO3 CYN FLC LCP NO2 NO3 OH OXL PER SCN SO3 or name O) and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+hits[0][0]+" and name "+ hits[0][3]+ " and resid "+str(hits[0][1])+")"))
                                    if len(res) == 1 and hit[1] in res:
                                        array.append(bonds[0])
                                    else:
                                        array.append(1000)
                                else:
                                    if hits[0][2] in ('SER','THR','TYR'):
                                        if hits[0][3] in ('OH','OG','OG1'):
                                            reverse_output = contactome.loc(axis=0)[hits[0][0],hits[0][1],hits[0][2],hits[0][3]+'d',:]
                                        else:
                                            res = np.unique(pqr.get('resid',sel="not resid "+str(hits[0][1])+" and ((name OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH and resname ASN ASP ASH GLN GLU GLH GLU HIP HID HIE SER THR TYR) or resname SO4 CL PO4 BR IOD ACT CO3 CYN FLC LCP NO2 NO3 OH OXL PER SCN SO3 or name O)  and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+hits[0][0]+" and name "+ hits[0][3]+ " and resid "+str(hits[0][1])+")"))
                                            if len(res) == 1 and hit[1] in res:
                                                array.append(bonds[0])
                                            else:
                                                array.append(1000)
                                        bonds=[]
                                        if len(reverse_output) == 1 and reverse_output.iloc[0,2]==his:
                                            array.append(reverse_output['bond'].iloc[0])
                                        elif len(reverse_output) > 1:
                                            if reverse_output['resname'].iloc[0] != his:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                                if substract >= 0.5:
                                                    array.append(1000)
                                                else:
                                                    array.append(5000)
                                            else:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                                if substract >= 0.5:
                                                    array.append(reverse_output['bond'].iloc[0])
                                                else:
                                                    array.append(1000)
                                    else:
                                        reverse_output = contactome.loc(axis=0)[hits[0]]
                                        if len(reverse_output) == 1 and reverse_output==['resname'].iloc[0]==his:
                                            array.append(reverse_output['bond'].iloc[0])
                                        elif len(reverse_output) > 1:
                                            if reverse_output.iloc[0,2] == his:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                                if substract >= 0.5:
                                                    array.append(reverse_output['bond'][0].iloc[0])
                                                else:
                                                    array.append(1000)
                                            elif reverse_output.iloc[0,2] != his:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.loc[reverse_output['resname']==his]['bond'].iloc[0])))
                                                if substract <= 0.5:
                                                    array.append(1000)
                                                else:
                                                    array.append(5000)
                            else:
                                decision_array=[]
                                bonds=[]
                                for num,data3 in data.groupby(level=4):
                                    hits.append((data3['chain'].iloc[0],float(data3['resid'].iloc[0]),data3['resname'].iloc[0],data3['name'].iloc[0]))
                                    bonds.append(data3['bond'].iloc[0])
                                for element in hits:
                                    if element[2] in ('SER','THR','TYR'):
                                        if element[3] in ('OH','OG','OG1'):
                                            reverse_output=contactome.loc(axis=0)[element[0],element[1],element[2],element[3]+'d',:]

                                        else:
                                            res = np.unique(pqr.get('resid',sel="not resid "+str(element[1])+" and ((name OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH and resname ASN ASP ASH GLN GLU GLH GLU HIP HID HIE SER THR TYR) or resname SO4 CL PO4 BR IOD ACT CO3 CYN FLC LCP NO2 NO3 OH OXL PER SCN SO3 or name O)  and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+element[0]+" and name "+ element[3]+ " and resid "+str(element[1])+")"))
                                            if len(res) == 1 and element[1] in res:
                                                decision_array.append(bonds[hits.index(element)])

                                            else:
                                                decision_array.append(1000)

                                        if len(reverse_output) == 1 and reverse_output.iloc[0,2]==his:
                                            decision_array.append(reverse_output['bond'].iloc[0])
                                        elif len(reverse_output) > 1:
                                            if reverse_output['resname'].iloc[0] != his:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                                if substract >= 0.5:
                                                    decision_array.append(1000)
                                                else:
                                                    decision_array.append(5000)
                                            else:
                                                substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                                if substract >= 0.5:
                                                    decision_array.append(reverse_output['bond'].iloc[0])
                                                else:
                                                    decision_array.append(1000)
                                    elif element not in contactome:
                                        res = np.unique(pqr.get('resid',sel="not resid "+str(element[1])+" and ((name OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH and resname ASN ASP ASH GLN GLU GLH GLU HIP HID HIE SER THR TYR) or resname SO4 CL PO4 BR IOD ACT CO3 CYN FLC LCP NO2 NO3 OH OXL PER SCN SO3 or name O)  and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+element[0]+" and name "+ element[3]+ " and resid "+str(element[1])+")"))
                                        if len(res) == 1 and element[1] in res:
                                            decision_array.append(bonds[hits.index(element)])
                                        else:
                                            decision_array.append(1000)
                                    else:
                                        reverse_output=(contactome.loc(axis=0)[element])
                                    if len(reverse_output) == 1 and reverse_output.iloc[0,2]==his:
                                        decision_array.append(reverse_output['bond'].iloc[0])
                                    elif len(reverse_output) > 1:
                                        if reverse_output['resname'].iloc[0] != his:
                                            substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                            if substract >= 0.5:
                                                decision_array.append(1000)
                                            else:
                                                decision_array.append(5000)
                                        else:
                                            substract=abs(float(reverse_output.iloc[0,4])-(float(reverse_output.iloc[1,4])))
                                            if substract >= 0.5:
                                                decision_array.append(reverse_output['bond'].iloc[0])
                                            else:
                                                decision_array.append(1000)
                                array.append(min(decision_array))

                #    print(hit,his)
                #    print(array,"\n") #UNCOMMENT TO SEE THE OUTPUT FROM THE HIS_TEST / INPUT HIS_JUDGE
                    predict = Contactome.his_judge(np.asarray(array),his)
                    if his not in predict:
                    #    sys.stderr.write("Dubious histidine resolution: | chain '%s' | resid '%s' | protonation '%s' || predicted --> %s \n"%(hit[0],hit[1],his," ".join(predict)))
                        his_dict[(str(hit[0]),str(hit[1]),str(his))] = tuple(predict)
        if len(his_dict) != 0:
            sys.stderr.write("DUBIOUS HISTIDINE:  (cutoff=%s)\n"%(cutoff))
            for key,value in his_dict.items():
                sys.stderr.write("Chain: %s | Resid: %s | Protonation: %s || Predicted --> %s \n"%(key[0],key[1],key[2]," ".join(value)))
        return(his_dict)

    def viewer(self,pqr,Chain,Resid,Name=None,displayChains=True):
        """Opens a VMD visualization with the contactome information for a given query

        Parameters
        ----------
        contactome : pandas.DataFrame.Contactome object
            The variable name where the Contatome object is saved
        pqr : str
            The variable name of the proteinPrepare result (pqr_object)
        Chain : str
            The Chain of the query you want to visualize
        Resid : int
            The Resid of the query you want to visualize
        Name: str
            The Name of the query you want to visualize. If None, it takes all the atoms avaiable in the Contactome query
        displayChains: bool
            A boolean to state whether the respective chains of the visualized residues should be displayed as well (in NewCartoon style). By default, this parameter is set to True
        """
        # COMPROVATIONS THAT THE INPUT INFORMATION IS RIGHT
        if Chain not in  self.index.levels[0]:
            sys.exit("ERROR: Chain %s not found within the structure.\n" %(Chain))
        elif Resid not in self.index.levels[1]:
            sys.exit("ERROR: Resid %s from chain %s not found within the possible residues making contacts.\n" %(Resid,Chain))
        if Name:
            name_list = []
            for i,j in self.loc(axis=0)[Chain,Resid,:,:,:].iterrows():
                name_list.append(i[3])
            if Name not in name_list:
                sys.exit("ERROR: Name %s not found in resid %s from chain %s." %(Name,Resid,Chain))
        # COMPROVATIONS MADE. NOW USING THE FUNCTION:
        hetatm = ('WAT','CA','K','MG','NA','ZN','FE','MN','CS','CU','HG','HO3','LI','NH4','NI','RB','RU','SO4','CL','PO4','BR','IOD','ACT','CO3','CYN','FLC','LCP','NO2','NO3','OH','OXL','PER','SCN','SO3')
        chain_list= list(np.unique(pqr.chain))# For some reason, vmd viewer does not keep original names of the chains
        r = range(len(chain_list)) # It renames chains as numbers from 0 to "n"
        numChain = r[chain_list.index(Chain)] # So we translate the chain name to its respective number
        htmd.config(viewer='VMD')
        pqr.view(sel="chain "+str(numChain)+" and resid "+str(Resid), style="Licorice", hold=True)
        if displayChains:
            pqr.view(sel="chain "+str(numChain), style="NewCartoon", color=9, hold=True)
        if Name:
            for i,j in self.loc(axis=0)[Chain,Resid,:,Name,:].iterrows():
                if i[4] > 0:
                    jChain = ""
                    if "d" in i[3] or "a" in i[3]:
                        pqr.view(sel="chain "+str(numChain)+" and resid "+str(Resid)+" and name "+Name[0:-1], style="Dotted", hold=True)
                    else:
                        pqr.view(sel="chain "+str(numChain)+" and resid "+str(Resid)+" and name "+Name, style="Dotted", hold=True)
                    if j[2] in hetatm:
                        jChain = (r[chain_list.index(j[0])])+1
                    else:
                        jChain = r[chain_list.index(j[0])]
                    if "d" in j[3] or "a" in j[3]:
                        pqr.view(sel="chain "+str(jChain)+" and resid "+str(j[1])+" and name "+j[3][0:-1], style="CPK", hold=True)
                    else:
                        pqr.view(sel="chain "+str(jChain)+" and resid "+str(j[1])+" and name "+j[3], style="CPK", hold=True)
                    pqr.view(sel="chain "+str(jChain)+" and resid "+str(j[1]), hold=True)
                    if displayChains:
                        pqr.view(sel="chain "+str(jChain), style="NewCartoon", color=12, hold=True)
        else:
            for i,j in self.loc(axis=0)[Chain,Resid,:,:,:].iterrows():
                if i[4] > 0:
                    jChain = ""
                    if "d" in i[3]  or "a" in i[3]:
                        pqr.view(sel="chain "+str(numChain)+" and resid "+str(Resid)+" and name "+i[3][0:-1], style="Dotted", hold=True)
                    else:
                        pqr.view(sel="chain "+str(numChain)+" and resid "+str(Resid)+" and name "+i[3], style="Dotted", hold=True)
                    if j[2] == "WAT":
                        jChain = (r[chain_list.index(j[0])])+1
                    else:
                        jChain = r[chain_list.index(j[0])]
                    if "d" in j[3] or "a" in j[3]:
                        pqr.view(sel="chain "+str(jChain)+" and resid "+str(j[1])+" and name "+j[3][0:-1], style="CPK", hold=True)
                    else:
                        pqr.view(sel="chain "+str(jChain)+" and resid "+str(j[1])+" and name "+j[3], style="CPK", hold=True)
                    pqr.view(sel="chain "+str(jChain)+" and resid "+str(j[1]), hold=True)
                    if displayChains:
                        pqr.view(sel="chain "+str(jChain), style="NewCartoon", color=12, hold=True)
        pqr.view()

class Aa_wiki(object):
    """Class used for the creation of the pandas.DataFrame.Contactome object"""
    acceptor_atoms = ('OD1', 'OD2', 'OE1', 'OE2', 'ND1','NE2','OG','OG1','OH')
    donor_atoms = ('OH','OG','OG1','NE','NH1','NH2','NZ','ND1','NE1','NE2','ND2')
    cations = ('CA','K','MG','NA','ZN','FE','MN','CS','CU','HG','HO3','LI','NH4','NI','RB','RU')
    anions = ('SO4','CL','PO4','BR','IOD','ACT','CO3','CYN','FLC','LCP','NO2','NO3','OH','OXL','PER','SCN','SO3')
    donor_dict = { #Tot hidrogens menys la hsitidina, que considero els heavy atoms #FALTA LYN
    'TYR':('OH',),
    'SER':('OG',),
    'THR':('OG1',),
    'TRP':('NE1',),
    'ARG':('NE','NH1','NH2'),
    'AR0':('NE','NH1','NH2'),
    'LYS':('NZ',),
    'LYN':('NZ',),
    'HID':('ND1','NE2'),
    'HIE':('ND1','NE2'),
    'HIP':('ND1','NE2'),
    'GLN':('NE2',),
    'ASN':('ND2',),
    }
    dpolar_name = { #Tot hidrogens menys la hsitidina, que considero els heavy atoms #FALTA LYN
    'TYR':'OHd',
    'SER':'OGd',
    'THR':'OG1d',
    }
    acceptor_dict = {
    'ASN':('OD1',),
    'ASP':('OD1','OD2'),
    'ASH':('OD1','OD2'), #Neutre de ASP (conveni)
    'GLN':('OE1',),
    'GLU':('OE1','OE2'),
    'GLH':('OE1','OE2'), #Neutre de GLU (conveni)
    'HIP':('ND1','NE2'), # Convenio ja que les histidines es miraran tant com acceptor com donor
    'HID':('ND1','NE2'),
    'HIE':('ND1','NE2'),
    'SER':('OG',),
    'THR':('OG1',),
    'TYR':('OH',)
    }
    def __init__(self,chain,resid,resname):
        self.chain = str(chain)
        self.resid = resid
        self.resname = str(resname.upper())

    def bond_distance(pqr,resid1,name1,chain1,resid2,name2,chain2):
        """Return a bond_distance between two given atoms.

        Parameters
        ----------
        pqr : str
            The variable name of the proteinPrepare result (pqr_object)
        resid1 : int
            residue number for the first atom
        name1 : str
            name for the first atom
        chain1 : str
            chain for the fist atom
        resid2 : int
            residue number for the second atom
        name2 : str
            name for the second atom
        chain2 : str
            chain for the second atom

        Returns
        -------
        bond distance :  float
        """
        a=pqr.get("coords","chain "+ str(chain1) + " and resid "+ str(resid1) + " and name " + str(name1))
        b=pqr.get("coords","chain "+ str(chain2) + " and resid "+ str(resid2) + " and name " + str(name2))
        return(round(np.linalg.norm(a-b),2))

    def contactome_maker(self,pqr,cutoff=3.5):
        """Return a dictionary with the contactome information of the contactome

        Parameters
        ----------

        pqr : str
            The variable name of the proteinPrepare result (pqr_object)
        cutoff : float
            Distance cutoff for looking the donor and acceptor atoms avaiable from the specific atom of the given aminoacid

        Returns
        -------
        his_dict :  dictionary
            Contains each specific atom for the given aminacid as a key and a tuple with the hit (target) information as a value
        """
        my_dict = {}
        switch = 0 #Permet diferenciar quin dels dos atoms de la HIS es treballa (suposant que estan en el mateix ordre!)
        if self.resname in Aa_wiki.donor_dict:
            for atom in Aa_wiki.donor_dict[self.resname]:
                res = pqr.get("resid", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.acceptor_atoms)+" and resname "+" ".join(Aa_wiki.acceptor_dict.keys())+ ") or resname "+ " ".join(Aa_wiki.anions)+" or name O) and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom + " and resid "+str(self.resid)+")")

                if len(res) != 0:
                    bond = []
                    name = pqr.get("resname", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.acceptor_atoms)+" and resname "+" ".join(Aa_wiki.acceptor_dict.keys())+ ") or resname "+ " ".join(Aa_wiki.anions)+" or name O) and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom + " and resid "+str(self.resid)+")")
                    target = pqr.get("name", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.acceptor_atoms)+" and resname "+" ".join(Aa_wiki.acceptor_dict.keys())+ ") or resname "+ " ".join(Aa_wiki.anions)+" or name O) and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom+ " and resid "+str(self.resid)+")")
                    chn = pqr.get("chain", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.acceptor_atoms)+" and resname "+" ".join(Aa_wiki.acceptor_dict.keys())+ ") or resname "+ " ".join(Aa_wiki.anions)+" or name O) and not(name NE2 and resname GLN) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom+ " and resid "+str(self.resid)+")")
                    for i in range(0,len(res)):
                        bond.append(Aa_wiki.bond_distance(pqr,self.resid,atom,self.chain,res[i],target[i],chn[i]))

                    if self.resname not in ('HID','HIE','HIP','TYR','SER','THR'):
                        value = np.asarray(list(zip(chn,res,name,target,bond)))
                        value = value[np.argsort(value[:, 4])]
                        count= 0
                        for hit in value:
                            count+=1
                            my_dict[(self.chain,self.resid,self.resname,atom,count)] = hit
                    elif self.resname in ('TYR','SER','THR'):
                            value = np.asarray(list(zip(chn,res,name,target,bond)))
                            value = value[np.argsort(value[:, 4])]
                            count= 0
                            for hit in value:
                                count+=1
                                my_dict[(self.chain,self.resid,self.resname,Aa_wiki.dpolar_name[self.resname],count)] = hit
                    elif self.resname in ('HID','HIE','HIP') and switch == 1:
                        value = np.asarray(list(zip(chn,res,name,target,bond)))
                        value = value[np.argsort(value[:, 4])]
                        count = 0
                        for hit in value:
                            count+=1
                            my_dict[(self.chain,self.resid,self.resname,'NE2d',count)] = hit
                        switch = 2
                    elif self.resname in ('HID','HIE','HIP') and switch == 0:
                        value = np.asarray(list(zip(chn,res,name,target,bond)))
                        value = value[np.argsort(value[:, 4])]
                        count = 0
                        for hit in value:
                            count += 1
                            my_dict[(self.chain,self.resid,self.resname,'ND1d',count)] = hit
                        switch = 1
                else:
                    if self.resname not in ('HID','HIE','HIP','TYR','SER','THR'):
                        my_dict[(self.chain,self.resid,self.resname,atom,0)] = ""
                    elif self.resname in ('TYR','SER','THR'):
                        my_dict[(self.chain,self.resid,self.resname,Aa_wiki.dpolar_name[self.resname],0)] = ""
                    elif self.resname in ('HID','HIE','HIP') and switch == 1:
                        my_dict[(self.chain,self.resid,self.resname,'NE2d',0)] = ""
                        switch = 2
                    elif self.resname in ('HID','HIE','HIP') and switch == 0:
                        my_dict[(self.chain,self.resid,self.resname,'ND1d',0)] = ""
                        switch = 1

                if switch == 2:
                    switch = 0

        if self.resname in Aa_wiki.acceptor_dict:
            for atom in Aa_wiki.acceptor_dict[self.resname]:
                res = pqr.get("resid", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.donor_atoms)+" and resname "+ " ".join(Aa_wiki.donor_dict.keys()) +") or resname "+ " ".join(Aa_wiki.cations)+" or name H) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom+ " and resid "+str(self.resid)+")")

                if len(res) != 0:
                    bond = []
                    name = pqr.get("resname", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.donor_atoms)+" and resname "+ " ".join(Aa_wiki.donor_dict.keys()) +") or resname "+ " ".join(Aa_wiki.cations)+" or name H) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom+ " and resid "+str(self.resid)+")")
                    target = pqr.get("name", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.donor_atoms)+" and resname "+ " ".join(Aa_wiki.donor_dict.keys()) +") or resname "+ " ".join(Aa_wiki.cations)+" or name H) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom+ " and resid "+str(self.resid)+")")
                    chn = pqr.get("chain", sel="not resid "+str(self.resid)+" and ((name "+" ".join(Aa_wiki.donor_atoms)+" and resname "+ " ".join(Aa_wiki.donor_dict.keys()) +") or resname "+ " ".join(Aa_wiki.cations)+" or name H) and within "+str(cutoff)+" of (chain "+self.chain+" and name "+ atom+ " and resid "+str(self.resid)+")")
                    for i in range(0,len(res)):
                        bond.append(Aa_wiki.bond_distance(pqr,self.resid,atom,self.chain,res[i],target[i],chn[i]))

                    if self.resname not in ('HID','HIE','HIP'):
                        value = np.asarray(list(zip(chn,res,name,target,bond)))
                        value = value[np.argsort(value[:, 4])]
                        count = 0
                        for hit in value:
                            count +=1
                            my_dict[(self.chain,self.resid,self.resname,atom,count)] = hit
                    elif self.resname in ('HID','HIE','HIP') and switch == 1:
                        value = np.asarray(list(zip(chn,res,name,target,bond)))
                        value = value[np.argsort(value[:, 4])]
                        count = 0
                        for hit in value:
                            count +=1
                            my_dict[(self.chain,self.resid,self.resname,'NE2a',count)] = hit
                        switch = 2
                    elif self.resname in ('HID','HIE','HIP') and switch == 0:
                        value = np.asarray(list(zip(chn,res,name,target,bond)))
                        value = value[np.argsort(value[:, 4])]
                        count = 0
                        for hit in value:
                            count +=1
                            my_dict[(self.chain,self.resid,self.resname,'ND1a',count)] = hit
                        switch = 1
                else:
                    if self.resname not in ('HID','HIE','HIP'):
                        my_dict[(self.chain,self.resid,self.resname,atom,0)] = ""
                    elif self.resname in ('HID','HIE','HIP') and switch == 1:
                        my_dict[(self.chain,self.resid,self.resname,'NE2a',0)] = ""
                        switch = 2
                    elif self.resname in ('HID','HIE','HIP') and switch == 0:
                        my_dict[(self.chain,self.resid,self.resname,'ND1a',0)] = ""
                        switch = 1

                if switch == 2:
                    switch = 0

        return my_dict
#----final class Aa_wiki----

#------ Contactome Maker ------
def contactome(pqr,prepData,cutoff=3.5,returnDetails=False):
    """Calls the contactome_maker and the check method and returns a Pandas.DataFrame.Contactome object

    Parameters
    ----------

    pqr : str
        The variable name of the proteinPrepare result (pqr_object)
    prepData : str
        The variable name of the proteinPrepare returnDetails = True
    cutoff : float
        Distance cutoff for looking the donor and acceptor atoms avaiable from the specific atom of the given aminoacid
    returnDetails: bool
        If True, it returns the pandas.DataFrame object of the check method.

    Returns
    -------
    pandas.DataFrame.Contactome : Pandas.DataFrame child class object
    """
    start = time.time()
    our_aa =  set(['HIP','LYS','ARG','HID','HIE','LYN','AR0','ASH','GLH','ASP','GLU','SER','THR','TYR','TYM','TRP','GLN','ASN'])
    chains = np.unique(pqr.get('chain'))
    my_dict = {}
    sys.stderr.write("\n**Contactome**: %s chains detected --> %s. Creating the contactome...\n"%(len(np.unique(prepData.data['chain']))," ".join(np.unique(prepData.data['chain']))))
    for index, row in prepData.data.iterrows():
        if row['protonation'] in our_aa:
            aa = Aa_wiki(row['chain'],row['resid'],row['protonation'])
            my_dict.update(aa.contactome_maker(pqr,cutoff=cutoff)) #Es pot posar un cutoff, per defecte =4
    ctctome = Contactome(my_dict)
    ctctome.rename(index={0: 'chain',1:'resid',2:'resname',3:'name',4:'bond'}, inplace=True)
    ctctome = ctctome.transpose()
    ctctome.index.names = ['Chain','Resid','Resname','Name','n']
    ctctome.loc[ctctome['resid'] != '','resid'] = ctctome.loc[ctctome['resid'] != '','resid'].astype(int)
    ctctome.loc[ctctome['bond'] != '','bond'] = ctctome.loc[ctctome['bond'] != '','bond'].astype(float)
    sys.stderr.write("Checking the contactome...\n")
    try:
        checking = ctctome.check(pqr,cutoff)
        if returnDetails:
            result = (ctctome,checking)
        else:
            result = ctctome
    except:
        sys.stderr.write("Some problem happened during the checking, returning just the contactome...")
        result = ctctome
    end = time.time()
    print("%.2f min"%((end-start)/60))
    return result
#----------------------------------
import time
