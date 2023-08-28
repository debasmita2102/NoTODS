from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit.providers.fake_provider import *
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit.compiler import transpile
from circuit_knitting_toolbox.circuit_cutting.wire_cutting import *
import mapomatic as mm

import numpy as np
import math

import sys
import cplex
import docplex.mp
from docplex.mp.model import Model

class NoTODS():

    def __init__(self,
                 circuit: QuantumCircuit,
                 hardware_list: list,
                 tau_list: list,
                ):
                    self.circuit = circuit
                    self.hardware_list = hardware_list
                    self.hardware_name = [x.backend_name for x in self.hardware_list]
                    self.tau_list = tau_list

                    self._set_cut_options()
                    self._set_opt_options()


    def _set_cut_options(self):
        self.max_cuts = 1
        self.num_subcircuits = [2]
        self.max_subcircuit_width = math.ceil(self.circuit.num_qubits/2)+1
        
        
    def _set_opt_options(self):
        self.t_2q = 10
        self.t_1q = 1


    def _cut_circuit(self) -> dict:
        cuts = cut_circuit_wires(
            circuit=self.circuit,
            method="automatic",
            max_subcircuit_width=self.max_subcircuit_width,
            max_cuts=self.max_cuts,
            num_subcircuits=self.num_subcircuits,
            verbose = False
        )
        
        return cuts

    
    def _get_valid_backends(self, cuts: dict) -> list:
        valid_backends = {}
        subcircuits = cuts['subcircuits']
        
        for idx, ckt in enumerate(subcircuits):
            valid_ckt_backends = []
            for b in self.hardware_list:
                if (b.configuration().n_qubits >=ckt.num_qubits):
                    valid_ckt_backends.append(b)

            valid_backends['subckt'+str(idx)] = valid_ckt_backends

        return valid_backends


    def _get_mm_score(self, cuts:dict, valid_backends:dict) -> list:
        layouts = []
        subcircuits = cuts['subcircuits']

        for idx, ckt in enumerate(subcircuits):
            trans_qc = transpile(ckt, valid_backends['subckt'+str(idx)][0], seed_transpiler=0, optimization_level=3)
            small_qc = mm.deflate_circuit(trans_qc)
            layout = mm.best_overall_layout(small_qc, valid_backends['subckt'+str(idx)], successors=True)
            layouts.append(layout)
            
        return layouts
    
    
    def _sort_mm_score(self, cuts:dict, layoutstructure:list) ->list:
        for backendname in self.hardware_name:
            flag = False

            for ckt_idx in range(len(cuts["subcircuits"])):
                flag = False
                for layout_idx in range(len(layoutstructure[ckt_idx])):
                    if backendname == layoutstructure[ckt_idx][layout_idx][1]:
                        flag = True
                        break
                if flag == False :
                    layoutstructure[ckt_idx].append((list(range(cuts["subcircuits"][ckt_idx].num_qubits)), backendname, 999))
                    
                    

        mapping_mapomatic= [0 for x in range(len(cuts["subcircuits"]))] 
        for i in range(len(cuts["subcircuits"])):
            mapping_mapomatic[i] = layoutstructure[i][0][0]
            
        score_mapomatic= []

        for ckt in range(len(cuts["subcircuits"])):
            score_mapomatic.append([])

        for i in range(len(cuts["subcircuits"])):
            for name in self.hardware_name:
                for entry in layoutstructure[i]:
                    if entry[1] == name:
                        score_mapomatic[i].append(entry[2])
                        
        return score_mapomatic
    
    
    
    def _eta_calculation(self , cuts: dict, score_mapomatic: list):
        score= [[0 for x in range(len(self.hardware_name))] for y in range(len(cuts["subcircuits"]))]
        eta_Array = []
        for i in range(len(cuts["subcircuits"])): #for each subckt
            #print("for subscircuit ", i)
            rho = cuts['counter'][i]['rho']
            o = cuts['counter'][i]['O']
            if rho!=0 and o!=0 :
                eta = rho*4*o*3
            elif rho == 0 :
                 eta = o*3
            elif o == 0 :
                eta = rho*4
            eta_Array.append(eta)



            for j in range (len(self.hardware_name)): #for each mach/ine    
                score[i][j] = eta * score_mapomatic[i][j] #subckt1 in machine 1, subckt1 in machine 2 and so on...
                
                
        return score, eta_Array
    
    
    
    
    
    def _optimization(self, cuts,score, eta_Array):
        m = Model(name='circuitcutting')
        time_array = [0 for x in range(len(cuts["subcircuits"]))]
   


        for i in range(len(cuts["subcircuits"])):
            time_array[i] =self.t_2q*cuts["subcircuits"][i].depth(lambda x: x[0].num_qubits == 2) + self.t_1q*cuts["subcircuits"][i].depth(lambda x: x[0].num_qubits == 1)

        F= [[0 for x in range(len(self.hardware_name))] for y in range(len(cuts["subcircuits"]))]

        #print(F) #for 0th subcircuit, all machine, for 1st subckt all machine and so on ...
        
        for i in range(len(cuts["subcircuits"])):
            for j in range(len(self.hardware_name)):

                F[i][j] = m.integer_var(0,1,name='F'+str(i)+'_'+str(j))

        sums = 0

  
        for i in range(len(cuts["subcircuits"])):
            for j in range(len(self.hardware_name)):
                sums += F[i][j]

            m.add_constraint(sums == 1)
            sums = 0

        sums = 0
        for j in range(len(self.hardware_name)):
            for i in range(len(cuts["subcircuits"])):

                sums += eta_Array[i]*time_array[i]*F[i][j]
            m.add_constraint(sums <= self.tau_list[j]) #add cosntrain for each machine time
            sums = 0

        sums = 0
        for i in range(len(cuts["subcircuits"])):
            for j in range(len(self.hardware_name)):
                sums += score[i][j]*F[i][j]

        m.minimize(sums)
        s = m.solve()
        m.print_solution()

        if s is None:
            print('- model is infeasible')

            
        dec_var = []
        sol = m.solution.as_dict()
        for key in sol.keys():
            dec_var.append(key.name)
        mapping = [[0 for x in range(len(cuts["subcircuits"]))] for y in range(len(self.hardware_name))]

        mapped_list = []
        for i in range(len(cuts["subcircuits"])):
            for j in range(len(self.hardware_name)):
                if 'F'+str(i)+'_'+str(j) in dec_var:

                    mapping[j][i]=1
                    mapped_list.append(self.hardware_name[j])
                    break
        return mapped_list
    
    
    
    
    def schedule(self):
        cuts  = self._cut_circuit()
        validlist = self._get_valid_backends(cuts)
        layout = self._get_mm_score(cuts, validlist)
        sortedlist = self._sort_mm_score(cuts,layout)
        score, eta_array = self._eta_calculation(cuts,sortedlist)
        model = self._optimization(cuts, score, eta_array)
        return model

 
                        




    



        