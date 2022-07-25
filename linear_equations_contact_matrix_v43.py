#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:24:32 2022

@author: ckadelka
"""

import numpy as np
import scipy.optimize as opti
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import itertools

'''This program projects age-structured social contact matrices to populations split not only by age but
by further Boolean attributes. For each Boolean attribute, the split across the population is known.
For attribute1, there may further be a non-random split across the population, i.e. homophily such that
there are more contacts among people with the same attribute value. This non-random split is described by
a scalar called homophily h in [0,1]:
the number of interactions between individuals with different attribute1 values is (1-h) times the expected
number (i.e. in the case h==0), where attribute 1 has no impact on social contact levels. 
In the other extreme case (h==1), there exists complete segregation: 
only individuals with the same attribute1 value interact.

Inputs:
    1. empirical age-age contact matrix (not necesarily "symmetric"), n_ages = number of age classes
    2. population sizes split by age (absolute numbers of proportion), i.e. a vector of length n_ages)
    3. proportion attribute1==True by age, i.e. a vector of length n_ages
    4. homophily h in [0,1] of attribute1 present in social contacts
    4. proportion attribute2==True by age and attribute1 status, i.e. an array of size (n_ages,2)
    5. proportion attribute3==True by age and attribute1 and attribute2 status, i.e. an array of size (n_ages,2,2)
    
        
'''

infix = 'method_paper_'


##Inputs:
C_old = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
people_per_age = np.array([ 60570126, 213610414,  31483433,  22574830])
p_attribute1 = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
homophily_attribute1 = 0
p_attribute2 = np.array([[1,1],[0.65,0.65],[1-0.79602454,1-0.79602454],[1-0.82338084,1-0.82338084]])
p_attribute3 = np.zeros((4,2,2))
p_attribute3[1] = np.array([[0.25,0.25],[0.4,0.4]])

C_old = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
N_old = np.array([ 60570126, 213610414,  31483433,  22574830])
p_attribute1 = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
homophily_attribute1 = 0
p_attribute2 = np.zeros((4,2))
p_attribute2[1,0] = 0.25
p_attribute2[1,1] = 0.5
p_attribute3 = np.ones((4,2,2))
p_attribute3[1] = 0.71
p_attribute3[2] = 0.282000012
p_attribute3[3] = 0.21699997



norm = 2 #norm used in objective function, p in [0,infinity), default = 2
#e.g., we want to find a 4x4 age-attribute1 contact matrix that satisfies these equations
#C[0,0] ... C[3,0] first column, ... C[3,0] ... C[3,3] last column
#in matrix form
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
# 12 13 14 15 


def plot_contact_matrix(C,filename): 
    f,ax = plt.subplots(figsize=(5,4))
    im=ax.imshow(np.log2(C),cmap=cm.Greens,vmin=np.log2(0.01),vmax=np.log2(16))   
    ax.set_yticks(range(18))
    ax.set_yticklabels(list(map(str,range(1,18+1))))
    ax.set_ylabel('Average daily contacts\n an individual in sub-population')
    ax.set_xticks(range(18))
    ax.set_xticklabels(list(map(str,range(1,18+1))))
    ax.set_xlabel('has with individuals of sub-population')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #f.subplots_adjust(left=0.1,right=0.7,bottom=0.2,top=0.9)
    #cbar_ax = f.add_axes([0.72, 0.212, 0.03, 0.6775])
    #cbar = f.colorbar(im, cax=cbar_ax)
    ##cbar = f.colorbar(im)
    #ticks = np.log2([0.01,0.1,1,5,10])
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
    #cbar.set_label('average daily contacts')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size="4%", pad=0.1)
    cbar=f.colorbar(im,cax=caxax)
    ticks = np.log2([0.01,0.1,1,10])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map(str,[0.01,0.1,1,10])))
    #cbar.set_label('average daily contacts')
    plt.savefig(filename,bbox_inches = "tight")

def plot_4x4_contact_matrix(C,filename): 
    f,ax=plt.subplots(figsize=(2.5,3))
    im=ax.imshow(np.log2(C),cmap=cm.Greens,vmin=np.log2(0.01),vmax=np.log2(16))    
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels(['0-15','16-64','65-74','75+'])
    ax.set_ylabel('Average daily contacts\nan individual of age')
    ax.set_xticks(range(len(C)))
    ax.set_xticklabels(['0-15','16-64','65-74','75+'])
    ax.set_xlabel('has with individuals of age')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #f.subplots_adjust(left=0.1,right=0.7,bottom=0.2,top=0.9)
    #cbar_ax = f.add_axes([0.72, 0.212, 0.03, 0.6775])
    #cbar = f.colorbar(im, cax=cbar_ax)
    ##cbar = f.colorbar(im)
    #ticks = np.log2([0.01,0.1,1,5,10])
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
    #cbar.set_label('average daily contacts')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size="8%", pad=0.1)
    cbar=f.colorbar(im,cax=caxax)
    ticks = np.log2([0.01,0.1,1,10])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map(str,[0.01,0.1,1,10])))
    #cbar.set_label('average daily contacts')
    plt.savefig(filename,bbox_inches = "tight")

def plot_8x8_contact_matrix(C,filename): 
    f,ax=plt.subplots(figsize=(2.5,3))
    im=ax.imshow(np.log2(C),cmap=cm.Greens,vmin=np.log2(0.01),vmax=np.log2(16))    
    ax.set_yticks(range(len(C)))
    age_labels = ['0-15','16-64','65-74','75+']
    ethnicity_labels = ['W or A','nW and nA']
    xlabels = [el1+' '+el2 for el1 in age_labels for el2 in ethnicity_labels]
    ylabels = [el1+' '+el2 for el1 in age_labels for el2 in ethnicity_labels]
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('Average daily contacts\nan individual of age')
    ax.set_xticks(range(len(C)))
    ax.set_xticklabels(xlabels,rotation=90)
    ax.set_xlabel('has with individuals of age',rotation=0)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #f.subplots_adjust(left=0.1,right=0.7,bottom=0.2,top=0.9)
    #cbar_ax = f.add_axes([0.72, 0.212, 0.03, 0.6775])
    #cbar = f.colorbar(im, cax=cbar_ax)
    ##cbar = f.colorbar(im)
    #ticks = np.log2([0.01,0.1,1,5,10])
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
    #cbar.set_label('average daily contacts')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size="8%", pad=0.1)
    cbar=f.colorbar(im,cax=caxax)
    ticks = np.log2([0.01,0.1,1,10])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map(str,[0.01,0.1,1,10])))
    #cbar.set_label('average daily contacts')
    plt.savefig(filename,bbox_inches = "tight")


class ContactMatrix:
    def __init__(self, C_old, N_old, prevalence_attribute, SAME_ROW_SUM=True, RECIPROCITY=True, HOMOPHILY=True, CORRECT_AGE_AGE_CONTACTS=True, norm=2):
        self.C_old = np.array(C_old)
        self.N_old = np.array(N_old)
        self.prevalence_attribute = prevalence_attribute
        
        self.SAME_ROW_SUM=SAME_ROW_SUM
        self.RECIPROCITY=RECIPROCITY
        self.HOMOPHILY=HOMOPHILY
        self.CORRECT_AGE_AGE_CONTACTS=CORRECT_AGE_AGE_CONTACTS
        self.norm = 2 #other p-norms don't work yet as the root of the p-norm is not implemented
        
        self.A_old = self.N_old.shape
        self.indices_old = list(itertools.product(*[list(range(dim)) for dim in self.A_old]))

        self.C_old_sim = self.symmetrize_Funk2020(self.C_old,self.N_old)
        
        assert self.prevalence_attribute.shape == self.A_old,'dimension mismatch'
        self.N = np.zeros((self.A_old+tuple([2])))
        for i in self.indices_old:
            self.N[i+tuple([1])] = self.N_old[i] * self.prevalence_attribute[i]
            self.N[i+tuple([0])] = self.N_old[i] * (1 - self.prevalence_attribute[i])
        self.A = self.N.shape
        self.indices = list(itertools.product(*[list(range(dim)) for dim in self.A]))
        
        self.particular_solution_h0 = self.get_particular_solution(homophily=0)
        self.particular_solution_h1 = self.get_particular_solution(homophily=1)



    def get_contact_matrix_when_homophily0(self):
        '''prevalence can either be of the shape as self.N or (*self.N,dim_x), 
        where dim_x is the number of different choices for the new attribute'''
        C = np.zeros((self.A+self.A))
        for i in self.indices_old:
            for j in self.indices_old:
                for v in range(2):
                    if self.N[i+tuple([v])] > 0:
                        C[i+tuple([v])+j+tuple([1])] = self.prevalence_attribute[j] * self.C_old_sim[i+j]
                        C[i+tuple([v])+j+tuple([0])] = (1-self.prevalence_attribute[j]) * self.C_old_sim[i+j]
        return C

    def get_number_of_opposite_attribute_interactions(self,C,N):
        number_of_opposite_attribute_interactions = 0
        for i in self.indices_old:
            for v in range(2):
                for j in self.indices_old:
                    number_of_opposite_attribute_interactions += N[i+tuple([v])] * C[i+tuple([v])+j+tuple([1-v])] / 2
        return number_of_opposite_attribute_interactions
    
    @staticmethod    
    def get_rowsums(C):
        n_dim = len(C.shape)
        return np.sum(C,tuple(range(n_dim//2,n_dim)))
    
    @staticmethod        
    def symmetrize_Funk2020(C,N):
        indices = list(itertools.product(*[list(range(dim)) for dim in N.shape]))
        return np.array([[1/2/N[i]*(C[(i+j)]*N[i]+C[(j+i)]*N[j]) if N[i]>0 else 0 for j in (indices)] for i in (indices)],dtype=np.float64)
    
    @staticmethod    
    def get_number_total_interactions(C,N):
        rowsums = ContactMatrix.get_rowsums(C)
        n_el = int(np.prod(N.shape))
        return np.dot(rowsums.reshape(n_el),N.reshape(n_el)) / 2

    @staticmethod
    def issymmetric(C,N,tol=1e-5):
        assert C.shape == N.shape+N.shape,'dimension mismatch'
        indices = list(itertools.product(*[list(range(dim)) for dim in N.shape]))
        for ii,i in enumerate(indices):
            for j in indices[ii+1:]:
                if np.abs(C[i+j]*N[i]-C[j+i]*N[j])>1e-5:
                    return False,i,j,C[i+j]*N[i],C[j+i]*N[j]
        return True   
    
    @staticmethod        
    def vec_to_2d_matrix(x): #as_matrix
        n_groups = int(np.sqrt(len(x)))
        return x.reshape(n_groups,n_groups)

    @staticmethod        
    def vec_to_nd_matrix(vec,shape): #as_matrix
        return vec.reshape(shape)
    
    
    @staticmethod        
    def nd_to_2d_matrix(C): #as_matrix
        n_groups = int(np.sqrt(np.prod(C.shape)))
        return C.reshape(n_groups,n_groups)    
    
    @staticmethod        
    def nd_matrix_to_vec(C): #as_matrix
        n_groups_squared = np.prod(C.shape)
        return C.reshape(n_groups_squared)  
    
    # @staticmethod        
    # def matrix_to_vec(matrix):
    #     n_groups = int(np.sqrt(len(x)))
    #     return x.reshape(n_groups,n_groups)    

    def get_equality_contraints_matrix(self,homophily=0,SAME_ROW_SUM=None,RECIPROCITY=None,HOMOPHILY=None,CORRECT_AGE_AGE_CONTACTS=None):
        LHS = []
        RHS = []
        if SAME_ROW_SUM or (SAME_ROW_SUM==None and self.SAME_ROW_SUM):
            #people of same age have same total number of contacts, except when the row corresponds to a group without population, which will be deleted at the end
            for i in self.indices:
                if self.N[i] > 0:
                    new_equation = np.zeros(self.A+self.A)
                    for j in self.indices:
                        new_equation[i+j] = 1
                    LHS.append(new_equation)
                    RHS.append(sum(self.C_old_sim[i[:-1]]))
                else:
                    for j in self.indices: #i has no contacts (i.e., empty row)
                        new_equation = np.zeros(self.A+self.A)
                        new_equation[i+j] = 1
                        LHS.append(new_equation)
                        RHS.append(0)
                        
                    for j in self.indices: #nobody has contacts with i (i.e., empty column)
                        new_equation = np.zeros(self.A+self.A)
                        new_equation[j+i] = 1
                        LHS.append(new_equation)
                        RHS.append(0)
        
        if RECIPROCITY  or (RECIPROCITY==None and self.RECIPROCITY):
            #direct RECIPROCITY argument
            for ii,i in enumerate(self.indices):
                for j in self.indices[ii+1:]:
                    new_equation = np.zeros(self.A+self.A)
                    new_equation[i+j] = self.N[i]
                    new_equation[j+i] = -self.N[j]
                    LHS.append(new_equation)
                    RHS.append(0)
                    
                
        if CORRECT_AGE_AGE_CONTACTS or (CORRECT_AGE_AGE_CONTACTS==None and self.CORRECT_AGE_AGE_CONTACTS):
            #each old group needs to have the correct contacts with each other old group (irrespective of the attribute)
            for i in self.indices_old:
                for j in self.indices_old:
                    new_equation = np.zeros(self.A+self.A)
                    new_equation[i+tuple([0])+j+tuple([0])] = 1-self.prevalence_attribute[i]
                    new_equation[i+tuple([0])+j+tuple([1])] = 1-self.prevalence_attribute[i]
                    new_equation[i+tuple([1])+j+tuple([0])] = self.prevalence_attribute[i]
                    new_equation[i+tuple([1])+j+tuple([1])] = self.prevalence_attribute[i]
                    LHS.append(new_equation)
                    RHS.append(self.C_old_sim[i+j])                    
                    
        if HOMOPHILY or (HOMOPHILY==None and self.HOMOPHILY):
            #given a value of homophily for attribute1
            #observed opposite-attribute1 interactions = homophily * #(opposite-attribute1 interactions without homophily)
            number_of_opposite_attribute_interactions_without_homophily = self.get_number_of_opposite_attribute_interactions(self.particular_solution_h0,self.N)
            
            new_equation = np.zeros(self.A+self.A)
            for i in self.indices_old:
                for v in range(2):
                    for j in self.indices_old:
                        new_equation[i+tuple([v])+j+tuple([1-v])] = self.N[i+tuple([v])] / 2
            LHS.append(new_equation)
            RHS.append((1-homophily) * number_of_opposite_attribute_interactions_without_homophily)

        if homophily==1: #if perfect segregation is desired
            for i in self.indices_old:
                for v in range(2):
                    for j in self.indices_old:
                        new_equation = np.zeros(self.A+self.A)
                        new_equation[i+tuple([v])+j+tuple([1-v])] = 1
                        LHS.append(new_equation)
                        RHS.append(0) 

        if homophily==-1: #if a bipartite network is desired
            for i in self.indices_old:
                for v in range(2):
                    for j in self.indices_old:
                        new_equation = np.zeros(self.A+self.A)
                        new_equation[i+tuple([v])+j+tuple([v])] = 1
                        LHS.append(new_equation)
                        RHS.append(0)
                        
        return (np.array(LHS),np.array(RHS)) 
    
    def get_null_space(self,homophily=0,SAME_ROW_SUM=None,RECIPROCITY=None,HOMOPHILY=None,CORRECT_AGE_AGE_CONTACTS=None):
        self.LHS,self.RHS = self.get_equality_contraints_matrix(homophily,SAME_ROW_SUM,RECIPROCITY,HOMOPHILY,CORRECT_AGE_AGE_CONTACTS)
        self.design_matrix = np.array([ContactMatrix.nd_matrix_to_vec(equation) for equation in self.LHS])
        self.null = scipy.linalg.null_space(self.design_matrix) #null space
        _,self.dim_null = self.null.shape        

    def get_particular_solution(self,homophily,SAME_ROW_SUM=None,RECIPROCITY=None,HOMOPHILY=None,CORRECT_AGE_AGE_CONTACTS=None):
        if homophily==0: #particular solution is trivial
            particular_solution = self.get_contact_matrix_when_homophily0()
            #particular_solution = np.reshape(particular_solution,np.prod(self.A)**2)
        else:
            self.get_null_space(homophily,SAME_ROW_SUM,RECIPROCITY,HOMOPHILY,CORRECT_AGE_AGE_CONTACTS)
            particular_solution = np.linalg.lstsq(self.design_matrix,self.RHS,rcond=None)[0]
            particular_solution = particular_solution.reshape(self.A+self.A)
        return particular_solution 

    def solve(self,homophily=0,tol=1e-6):
        self.homophily = homophily
        self.get_null_space(homophily)
        self.particular_solution = homophily*self.particular_solution_h1 + (1-homophily)*self.particular_solution_h0#np.linalg.lstsq(A,b,rcond=None)[0]
        self.n_groups = int(np.prod(self.A))
        self.particular_solution = self.particular_solution.reshape(self.n_groups**2)
        
        linear_constraint = opti.LinearConstraint(self.null, lb = -self.particular_solution-tol,ub=np.inf*np.ones(self.n_groups**2)) #need to subtract tol due to float imprecision
        bounds = opti.Bounds(-np.inf*np.ones(self.dim_null), np.inf*np.ones(self.dim_null))
        x0 = np.zeros(self.dim_null) #x0==0 --> initial guess = particular solution
        self.result_optimization = opti.minimize(self.obj_func2, x0, jac=self.obj_func2_jac,method='SLSQP', tol=1e-35,constraints=[linear_constraint, ], bounds=bounds)
        x = np.dot(self.null,self.result_optimization.x)+self.particular_solution
        return x.reshape(self.A+self.A)#self.adjust_for_multiplier(x)   
     
    def obj_func1(self,param):
        x = np.dot(self.null,param)+self.particular_solution
        return sum(list(map(lambda el: el**self.norm,filter(lambda v: v==v, self.get_balance(x).flatten()))))    
    
    def get_balance(self,x):
        C = np.reshape(x,self.A+self.A)    
        difference_age_age_contacts = np.nan*np.ones(self.A_old+self.A_old)
        for i in self.indices_old:
            for j in self.indices_old:
                difference_age_age_contacts[i+j] = (C[i+tuple([1])+j+tuple([1])] + C[i+tuple([1])+j+tuple([0])] - C[i+tuple([0])+j+tuple([1])] - C[i+tuple([0])+j+tuple([0])])
        return difference_age_age_contacts
    
    def obj_func2(self,param):
        x = np.dot(self.null,param)+self.particular_solution
        return sum(list(map(lambda el: (el-self.homophily)**self.norm,filter(lambda v: v==v, self.get_specific_homophily_values(x).flatten()))))

    def obj_func2_jac(self,param):
        x = np.dot(self.null,param)+self.particular_solution
        C = np.reshape(x,self.A+self.A)    
        gradient = np.zeros(self.A+self.A)
        for i in self.indices_old:
            for j in self.indices_old:
                if self.prevalence_attribute[j] in [0,1]:#special case: if we have a subpopulation that does not split with the attribute
                    pass #gradient is zero
                else:
                    for v in range(2):#compute gradient
                        if self.N[i+tuple([v])]>0: #else pass
                            prop = 1-self.prevalence_attribute[j] if v==1 else self.prevalence_attribute[j]
                            sum_Civj = C[i+tuple([v])+j+tuple([0])] + C[i+tuple([v])+j+tuple([1])]  
                            
                            inside_derivative = (+1) * C[i+tuple([v])+j+tuple([1-v])] / sum_Civj**2 / prop
                            gradient[i+tuple([v])+j+tuple([v])] = self.norm*(1 - C[i+tuple([v])+j+tuple([1-v])] / sum_Civj / prop - self.homophily)**(self.norm-1) * inside_derivative
    
                            inside_derivative = (-1) * C[i+tuple([v])+j+tuple([v])] / sum_Civj**2 / prop
                            gradient[i+tuple([v])+j+tuple([1-v])] = self.norm*(1 - C[i+tuple([v])+j+tuple([1-v])] / sum_Civj / prop - self.homophily)**(self.norm-1) * inside_derivative
                            
        return np.dot(gradient.flatten(),self.null)   

    def get_specific_homophily_values(self,x):
        C = np.reshape(x,self.A+self.A)    
        specific_homophily_values = np.nan*np.ones(self.A+self.A_old)
        for i in self.indices_old:
            for j in self.indices_old:
                if self.prevalence_attribute[j] in [0,1]:
                    pass #homophily cannot be defined here, keep nan values
                else:
                    for v in range(2):
                        if self.N[i+tuple([v])]>0: #else keep nan values
                            prop = 1-self.prevalence_attribute[j] if v==1 else self.prevalence_attribute[j]
                            specific_homophily_values[i+tuple([v])+j] = 1 - C[i+tuple([v])+j+tuple([1-v])] / (C[i+tuple([v])+j+tuple([0])] + C[i+tuple([v])+j+tuple([1])]) / prop
        return specific_homophily_values

    def delete_empty_populations(self,C):
        indices_out = []
        N_out = []
        for i in self.indices:
            if self.N[i] > 0:
                indices_out.append(i)
                N_out.append(self.N[i])
        C_out = np.zeros((len(N_out),len(N_out)))
        for ii,i in enumerate(indices_out):
            for jj,j in enumerate(indices_out):
                C_out[ii,jj] = C[i+j]
        N_out = np.array(N_out)
        indices_out = np.array(indices_out)
        return (C_out,N_out,indices_out)    
    
# C = np.array([[9,12],[4,15]])
# N = np.array([100,300])
# prevalence_attribute = np.array([0.5,0.8])
# m = ContactMatrix(C,N,prevalence_attribute)
# print(m.issymmetric(C,N))
# S=m.symmetrize_Funk2020(C,N)
# print(m.issymmetric(S,N))
# CC = m.get_contact_matrix_when_homophily0()
# (LHS,RHS) = m.get_equality_contraints_matrix(0)
# contact_matrix = m.solve(0.5)
# print(m.get_specific_homophily_values(contact_matrix))
# print(m.issymmetric(contact_matrix,m.N))


    
# C = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
# N = np.array([ 60570126, 213610414,  31483433,  22574830])
# prevalence_attribute = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
# m = ContactMatrix(C,N,prevalence_attribute)
# print(m.issymmetric(C,N))
# S=m.symmetrize_Funk2020(C,N)
# print(m.issymmetric(S,N))
# CC = m.get_contact_matrix_when_homophily0()
# (LHS,RHS) = m.get_equality_contraints_matrix(0)
# contact_matrix = m.solve(0.5)
# print(m.get_specific_homophily_values(contact_matrix))
# print(m.issymmetric(contact_matrix,m.N))



    
class ContactMatrix2:
    def __init__(self, C_old, N_old, N_new, homophily_attribute1, multipliers_attribute2, SAME_ROW_SUM=True, RECIPROCITY=True, HOMOPHILY=True, CORRECT_AGE_AGE_CONTACTS=True, norm=2):
        self.C_old = np.array(C_old)
        self.N_old = np.array(N_old)
        self.N = N_new
        self.homophily_attribute1 = homophily_attribute1
        self.multipliers_attribute2 = multipliers_attribute2
        
        
        self.SAME_ROW_SUM=SAME_ROW_SUM
        self.RECIPROCITY=RECIPROCITY
        self.HOMOPHILY=HOMOPHILY
        self.CORRECT_AGE_AGE_CONTACTS=CORRECT_AGE_AGE_CONTACTS
        self.norm = 2 #other p-norms don't work yet as the root of the p-norm is not implemented
        
        self.A_old = self.N_old.shape
        self.indices_old = list(itertools.product(*[list(range(dim)) for dim in self.A_old]))

        self.C_old_sim = self.symmetrize_Funk2020(self.C_old,self.N_old)
        
        self.prevalence_new_attributes_given_old_index = np.array([N_new[i]/np.sum(N_new[i]) for i in self.indices_old])
        self.N_times_multipler = np.array([np.multiply(N_new[i],self.multipliers_attribute2) for i in self.indices_old])
        self.relative_contact_distribution_given_old_index = np.array([self.N_times_multipler[i] / np.sum(self.N_times_multipler[i]) for i in self.indices_old])
        self.desired_rowsum_multiplier_without_multiplier = np.array([np.sum(N_new[i])/np.sum(self.N_times_multipler[i]) for i in self.indices_old])
        
        #assert self.prevalence_attribute.shape == self.A_old,'dimension mismatch'
        #self.N = np.zeros((self.A_old+tuple([2])))
        #for i in self.indices_old:
        #    self.N[i+tuple([1])] = self.N_old[i] * self.prevalence_attribute[i]
        #    self.N[i+tuple([0])] = self.N_old[i] * (1 - self.prevalence_attribute[i])
        self.A = self.N.shape
        self.indices = list(itertools.product(*[list(range(dim)) for dim in self.A]))
       # 
        self.particular_solution_h0 = self.get_particular_solution(homophily=0)
        self.particular_solution_h1 = self.get_particular_solution(homophily=1)

    def get_contact_matrix_when_homophily0(self):
        '''prevalence can either be of the shape as self.N or (*self.N,dim_x), 
        where dim_x is the number of different choices for the new attribute'''
        C = np.zeros((self.A+self.A))
        for i in self.indices_old:
            for v in range(2):
                for w in range(2):
                    if self.N[i+tuple([v,w])] > 0:
                        for j in self.indices_old:
                            for vv in range(2):
                                for ww in range(2):
                                    C[i+tuple([v,w])+j+tuple([vv,ww])] = (self.relative_contact_distribution_given_old_index[j,vv,ww] * 
                                                                          self.C_old_sim[i+j] * 
                                                                          self.desired_rowsum_multiplier_without_multiplier[i] * 
                                                                          self.multipliers_attribute2[w])
        return C

    def get_number_of_opposite_attribute_interactions(self,C,N):
        number_of_opposite_attribute_interactions = 0
        for i in self.indices_old:
            for v in range(2):
                for w in range(2):
                    for j in self.indices_old:
                        for ww in range(2):
                            number_of_opposite_attribute_interactions += N[i+tuple([v,w])] * C[i+tuple([v,w])+j+tuple([1-v,ww])] / 2
        return number_of_opposite_attribute_interactions
    
    @staticmethod    
    def get_rowsums(C):
        n_dim = len(C.shape)
        return np.sum(C,tuple(range(n_dim//2,n_dim)))
    
    @staticmethod        
    def symmetrize_Funk2020(C,N):
        indices = list(itertools.product(*[list(range(dim)) for dim in N.shape]))
        return np.array([[1/2/N[i]*(C[(i+j)]*N[i]+C[(j+i)]*N[j]) if N[i]>0 else 0 for j in (indices)] for i in (indices)],dtype=np.float64)
    
    @staticmethod    
    def get_number_total_interactions(C,N):
        rowsums = ContactMatrix.get_rowsums(C)
        n_el = int(np.prod(N.shape))
        return np.dot(rowsums.reshape(n_el),N.reshape(n_el)) / 2

    @staticmethod
    def issymmetric(C,N,tol=1e-5):
        assert C.shape == N.shape+N.shape,'dimension mismatch'
        indices = list(itertools.product(*[list(range(dim)) for dim in N.shape]))
        for ii,i in enumerate(indices):
            for j in indices[ii+1:]:
                if np.abs(C[i+j]*N[i]-C[j+i]*N[j])>1e-5:
                    return False,i,j,C[i+j]*N[i],C[j+i]*N[j]
        return True   
    
    @staticmethod        
    def vec_to_2d_matrix(x): #as_matrix
        n_groups = int(np.sqrt(len(x)))
        return x.reshape(n_groups,n_groups)

    @staticmethod        
    def vec_to_nd_matrix(vec,shape): #as_matrix
        return vec.reshape(shape)
    
    
    @staticmethod        
    def nd_to_2d_matrix(C): #as_matrix
        n_groups = int(np.sqrt(np.prod(C.shape)))
        return C.reshape(n_groups,n_groups)    
    
    @staticmethod        
    def nd_matrix_to_vec(C): #as_matrix
        n_groups_squared = np.prod(C.shape)
        return C.reshape(n_groups_squared)  
    
    # @staticmethod        
    # def matrix_to_vec(matrix):
    #     n_groups = int(np.sqrt(len(x)))
    #     return x.reshape(n_groups,n_groups)    

    def get_equality_contraints_matrix(self,homophily=0,SAME_ROW_SUM=None,RECIPROCITY=None,HOMOPHILY=None,CORRECT_AGE_AGE_CONTACTS=None):
        LHS = []
        RHS = []
        if SAME_ROW_SUM or (SAME_ROW_SUM==None and self.SAME_ROW_SUM):
            #people of same age have same total number of contacts, except when the row corresponds to a group without population, which will be deleted at the end
            for i in self.indices:
                if self.N[i] > 0:
                    i_old = i[:-2]
                    w = i[-1]
                    desired_average_contacts = (np.sum(self.C_old_sim[i_old]) * 
                                                self.desired_rowsum_multiplier_without_multiplier[i_old] * 
                                                self.multipliers_attribute2[w])
                    new_equation = np.zeros(self.A+self.A)
                    for j in self.indices:
                        new_equation[i+j] = 1
                    LHS.append(new_equation)
                    RHS.append(desired_average_contacts)
                else:
                    for j in self.indices: #i has no contacts (i.e., empty row)
                        new_equation = np.zeros(self.A+self.A)
                        new_equation[i+j] = 1
                        LHS.append(new_equation)
                        RHS.append(0)
                        
                    for j in self.indices: #nobody has contacts with i (i.e., empty column)
                        new_equation = np.zeros(self.A+self.A)
                        new_equation[j+i] = 1
                        LHS.append(new_equation)
                        RHS.append(0)
        
        if RECIPROCITY  or (RECIPROCITY==None and self.RECIPROCITY):
            #direct RECIPROCITY argument
            for ii,i in enumerate(self.indices):
                for j in self.indices[ii+1:]:
                    new_equation = np.zeros(self.A+self.A)
                    new_equation[i+j] = self.N[i]
                    new_equation[j+i] = -self.N[j]
                    LHS.append(new_equation)
                    RHS.append(0)
        
        #exact multiple of contacts
        for i in self.indices_old:
            for v in range(2):
                if self.N[i+tuple([v,0])] > 0 and self.N[i+tuple([v,1])] > 0:
                    for j in self.indices_old:   
                        for vv in range(2):
                            for ww in range(2):
                                new_equation = np.zeros(self.A+self.A)
                                new_equation[i+tuple([v,0])+j+tuple([vv,ww])] = self.multipliers_attribute2[1]
                                new_equation[i+tuple([v,1])+j+tuple([vv,ww])] = -self.multipliers_attribute2[0]
                                LHS.append(new_equation)
                                RHS.append(0)                            
                            
        if CORRECT_AGE_AGE_CONTACTS or (CORRECT_AGE_AGE_CONTACTS==None and self.CORRECT_AGE_AGE_CONTACTS):
            #each old group needs to have the correct contacts with each other old group (irrespective of the attribute)
            for i in self.indices_old:
                for j in self.indices_old:
                    new_equation = np.zeros(self.A+self.A)
                    for v,w,vv,ww in list(itertools.product(range(2),repeat=4)):
                        new_equation[i+tuple([v,w])+j+tuple([vv,ww])] = self.prevalence_new_attributes_given_old_index[i,v,w]
                    LHS.append(new_equation)
                    RHS.append(self.C_old_sim[i+j])                    
                    
        if HOMOPHILY or (HOMOPHILY==None and self.HOMOPHILY):
            #given a value of homophily for attribute1
            #observed opposite-attribute1 interactions = homophily * #(opposite-attribute1 interactions without homophily)
            number_of_opposite_attribute_interactions_without_homophily = self.get_number_of_opposite_attribute_interactions(self.particular_solution_h0,self.N)
            
            new_equation = np.zeros(self.A+self.A)
            for i in self.indices_old:
                for v in range(2):
                    for w in range(2):
                        for j in self.indices_old:
                            for ww in range(2):
                                new_equation[i+tuple([v,w])+j+tuple([1-v,ww])] = self.N[i+tuple([v,w])] / 2
            LHS.append(new_equation)
            RHS.append((1-homophily) * number_of_opposite_attribute_interactions_without_homophily)

        if homophily==1: #if perfect segregation is desired
            for i in self.indices_old:
                for v in range(2):
                    for w in range(2):
                        for j in self.indices_old:
                            for ww in range(2):
                                new_equation = np.zeros(self.A+self.A)
                                new_equation[i+tuple([v,w])+j+tuple([1-v,ww])] = 1
                                LHS.append(new_equation)
                                RHS.append(0) 

        if homophily==-1: #if a bipartite network is desired
            for i in self.indices_old:
                for v in range(2):
                    for w in range(2):
                        for j in self.indices_old:
                            for ww in range(2):
                                new_equation = np.zeros(self.A+self.A)
                                new_equation[i+tuple([v,w])+j+tuple([v,ww])] = 1
                                LHS.append(new_equation)
                                RHS.append(0) 
                        
        return (np.array(LHS),np.array(RHS))
    
    def get_null_space(self,homophily=0,SAME_ROW_SUM=None,RECIPROCITY=None,HOMOPHILY=None,CORRECT_AGE_AGE_CONTACTS=None):
        self.LHS,self.RHS = self.get_equality_contraints_matrix(homophily,SAME_ROW_SUM,RECIPROCITY,HOMOPHILY,CORRECT_AGE_AGE_CONTACTS)
        self.design_matrix = np.array([ContactMatrix.nd_matrix_to_vec(equation) for equation in self.LHS])
        self.null = scipy.linalg.null_space(self.design_matrix) #null space
        _,self.dim_null = self.null.shape        

    def get_particular_solution(self,homophily,SAME_ROW_SUM=None,RECIPROCITY=None,HOMOPHILY=None,CORRECT_AGE_AGE_CONTACTS=None):
        if homophily==0: #particular solution is trivial
            particular_solution = self.get_contact_matrix_when_homophily0()
            #particular_solution = np.reshape(particular_solution,np.prod(self.A)**2)
        else:
            self.get_null_space(homophily,SAME_ROW_SUM,RECIPROCITY,HOMOPHILY,CORRECT_AGE_AGE_CONTACTS)
            particular_solution = np.linalg.lstsq(self.design_matrix,self.RHS,rcond=None)[0]
            particular_solution = particular_solution.reshape(self.A+self.A)
        return particular_solution   

    def solve(self,homophily=0,tol=1e-6):
        self.homophily = homophily
        self.get_null_space(homophily)
        self.particular_solution = homophily*self.particular_solution_h1 + (1-homophily)*self.particular_solution_h0#np.linalg.lstsq(A,b,rcond=None)[0]
        self.n_groups = int(np.prod(self.A))
        self.particular_solution = self.particular_solution.reshape(self.n_groups**2)
        
        linear_constraint = opti.LinearConstraint(self.null, lb = -self.particular_solution-tol,ub=np.inf*np.ones(self.n_groups**2)) #need to subtract tol due to float imprecision
        bounds = opti.Bounds(-np.inf*np.ones(self.dim_null), np.inf*np.ones(self.dim_null))
        x0 = np.zeros(self.dim_null) #x0==0 --> initial guess = particular solution
        self.result_optimization = opti.minimize(self.obj_func2, x0, jac=self.obj_func2_jac,method='SLSQP', tol=1e-35,constraints=[linear_constraint, ], bounds=bounds)
        x = np.dot(self.null,self.result_optimization.x)+self.particular_solution
        return x.reshape(self.A+self.A)#self.adjust_for_multiplier(x)   
     
    def obj_func1(self,param):
        x = np.dot(self.null,param)+self.particular_solution
        return sum(list(map(lambda el: (el-self.homophily)**self.norm,filter(lambda v: v==v, self.get_specific_homophily_values(x).flatten()))))

    def obj_func2(self,param):
        x = np.dot(self.null,param)+self.particular_solution
        return sum(list(map(lambda el: (el-self.homophily)**self.norm,filter(lambda v: v==v, self.get_specific_homophily_values(x).flatten()))))

    def obj_func2_jac(self,param):
        x = np.dot(self.null,param)+self.particular_solution
        C = np.reshape(x,self.A+self.A)    
        gradient = np.zeros(self.A+self.A)
        for i in self.indices_old:
            for j in self.indices_old:
                for ww in range(2):
                    if (self.prevalence_new_attributes_given_old_index[j,0,ww] in [0,1] and 
                        self.prevalence_new_attributes_given_old_index[j,1,ww] in [0,1]):
                        pass #homophily cannot be defined here, keep nan values
                    else:
                        for v in range(2):
                            for w in range(2):
                                if self.N[i+tuple([v,w])]>0: #else keep nan values
                                    #prop = 1-self.prevalence_attribute[j] if v==1 else self.prevalence_attribute[j]
                                    prop = self.relative_contact_distribution_given_old_index[j+tuple([1-v,ww])] / (self.relative_contact_distribution_given_old_index[j+tuple([0,ww])] + self.relative_contact_distribution_given_old_index[j+tuple([1,ww])])

                                    sum_Civj = C[i+tuple([v,w])+j+tuple([0,ww])] + C[i+tuple([v,w])+j+tuple([1,ww])]  
                                    
                                    inside_derivative = (+1) * C[i+tuple([v,w])+j+tuple([1-v,ww])] / sum_Civj**2 / prop
                                    gradient[i+tuple([v,w])+j+tuple([v,ww])] = self.norm*(1 - C[i+tuple([v,w])+j+tuple([1-v,ww])] / sum_Civj / prop - self.homophily)**(self.norm-1) * inside_derivative
            
                                    inside_derivative = (-1) * C[i+tuple([v,w])+j+tuple([v,ww])] / sum_Civj**2 / prop
                                    gradient[i+tuple([v,w])+j+tuple([1-v,ww])] = self.norm*(1 - C[i+tuple([v,w])+j+tuple([1-v,ww])] / sum_Civj / prop - self.homophily)**(self.norm-1) * inside_derivative
                                    
        return np.dot(gradient.flatten(),self.null)   

    def get_specific_homophily_values(self,x):
        C = np.reshape(x,self.A+self.A)    
        specific_homophily_values = np.nan*np.ones(self.A+self.A_old+tuple([2]))
        for i in self.indices_old:
            for j in self.indices_old:
                for ww in range(2):
                    if (self.prevalence_new_attributes_given_old_index[j,0,ww] in [0,1] and 
                        self.prevalence_new_attributes_given_old_index[j,1,ww] in [0,1]):
                        pass #homophily cannot be defined here, keep nan values
                    else:
                        for v in range(2):
                            for w in range(2):
                                if self.N[i+tuple([v,w])]>0: #else keep nan values
                                    #prop = 1-self.prevalence_attribute[j] if v==1 else self.prevalence_attribute[j]
                                    prop = self.relative_contact_distribution_given_old_index[j+tuple([1-v,ww])] / (self.relative_contact_distribution_given_old_index[j+tuple([0,ww])] + self.relative_contact_distribution_given_old_index[j+tuple([1,ww])])
                                    specific_homophily_values[i+tuple([v,w])+j+tuple([ww])] = 1 - C[i+tuple([v,w])+j+tuple([1-v,ww])] / (C[i+tuple([v,w])+j+tuple([0,ww])] + C[i+tuple([v,w])+j+tuple([1,ww])]) / prop
        return specific_homophily_values

    def delete_empty_populations(self,C):
        indices_out = []
        N_out = []
        for i in self.indices:
            if self.N[i] > 0:
                indices_out.append(i)
                N_out.append(self.N[i])
        C_out = np.zeros((len(N_out),len(N_out)))
        for ii,i in enumerate(indices_out):
            for jj,j in enumerate(indices_out):
                C_out[ii,jj] = C[i+j]
        N_out = np.array(N_out)
        indices_out = np.array(indices_out)
        return (C_out,N_out,indices_out)


def plot_contact_matrix(C_out,N_out,indices_out,names_attributes,filename,tol=1e-5): 
    len_N = len(N_out)
    f,ax = plt.subplots(figsize=(4,3))
    im=ax.imshow(np.log2(C_out),cmap=cm.Greens,vmin=np.log2(0.01),vmax=np.log2(16))   
    ax.set_yticks(range(len_N))
    ax.set_yticklabels(list(map(str,range(1,len_N+1))))
    ax.set_ylabel('Average daily contacts\n an individual in sub-population')
    ax.set_xticks(range(len_N))
    ax.set_xticklabels(list(map(str,range(1,len_N+1))))
    ax.set_xlabel('has with individuals of sub-population')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #f.subplots_adjust(left=0.1,right=0.7,bottom=0.2,top=0.9)
    #cbar_ax = f.add_axes([0.72, 0.212, 0.03, 0.6775])
    #cbar = f.colorbar(im, cax=cbar_ax)
    ##cbar = f.colorbar(im)
    #ticks = np.log2([0.01,0.1,1,5,10])
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
    #cbar.set_label('average daily contacts')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size="4%", pad=0.1)
    cbar=f.colorbar(im,cax=caxax)
    ticks = np.log2([0.01,0.1,1,10])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map(str,[0.01,0.1,1,10])))
    #cbar.set_label('average daily contacts')
    plt.savefig(filename,bbox_inches = "tight")

def plot_contact_matrix_specific(C_out,N_out,indices_out,names_attributes,filename): 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    len_N = len(N_out)
    names = [' '.join([names_attributes[ii][index] for ii,index in enumerate(indices)]) for ii,indices in enumerate(indices_out)]
    for i in range(len_N):
        if '16' not in names[i]:
            names[i] = names[i].replace(' LC','')

    f,ax = plt.subplots(figsize=(4,3))
    im=ax.imshow(np.log2(C_out),cmap=cm.Greens,vmin=np.log2(0.01),vmax=np.log2(16))   
    ax.set_yticks(range(len_N))
    ax.set_yticklabels(list(map(str,range(1,len_N+1))))
    ax.set_yticklabels(names)
    ax.set_ylabel('Average daily contacts\n an individual in sub-population')
    ax.set_ylabel('Average daily contacts\n a person who is')
    ax.set_xticks(range(len_N))
    ax.set_xticklabels(list(map(str,range(1,len_N+1))))
    ax.set_xticklabels(names,rotation=90)
    ax.set_xlabel('has with individuals of sub-population')
    ax.set_xlabel('has with people who are')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #f.subplots_adjust(left=0.1,right=0.7,bottom=0.2,top=0.9)
    #cbar_ax = f.add_axes([0.72, 0.212, 0.03, 0.6775])
    #cbar = f.colorbar(im, cax=cbar_ax)
    ##cbar = f.colorbar(im)
    #ticks = np.log2([0.01,0.1,1,5,10])
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
    #cbar.set_label('average daily contacts')
    
    #for i in range(len_N):
   #     ax.text(len_N+4,i,str(i+1)+' = '+names[i],va='center',ha='left')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size="4%", pad=0.1)
    cbar=f.colorbar(im,cax=caxax)
    ticks = np.log2([0.01,0.1,1,10])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map(str,[0.01,0.1,1,10])))
    #cbar.set_label('average daily contacts')
    plt.savefig(filename,bbox_inches = "tight")

if __name__ == '__main__':
    #Example data
    C = np.array([[9,12],[4,15]])
    N = np.array([100,300])
    prevalence_attribute = np.array([0.01,0.99])
    m = ContactMatrix(C,N,prevalence_attribute)
    print(m.issymmetric(C,N))
    S=m.symmetrize_Funk2020(C,N)
    print(m.issymmetric(S,N))
    CC_small = m.get_contact_matrix_when_homophily0()
    (LHS,RHS) = m.get_equality_contraints_matrix(0)
    contact_matrix = m.solve(0.5)
    print(m.get_specific_homophily_values(contact_matrix))
    print(m.issymmetric(contact_matrix,m.N))


    C = np.array([[9,12],[4,15]])
    N = np.array([100,300])
    #N_new = np.array([[[25,25],[20,30]] , [[30,30],[180,60]]])
    N_new = np.array([[[50,0],[50,0]] , [[30,30],[180,60]]])
    N_new = np.array([[[50,0],[50,0]] , [[60,0],[240,0]]])
    homophily_attribute1 = 1
    multipliers_attribute2 = [1,2]
    m = ContactMatrix2(C,N,N_new,homophily_attribute1,multipliers_attribute2)
    CC = m.solve(1)





# #Real U.S. data
# empirical_contact_matrix = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
# census_data = np.array([ 60570126, 213610414,  31483433,  22574830])
# prevalence_WorA_ethnicity = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
# prevalence_high_contact = np.array([[0,0],[0.5,0.25],[0,0],[0,0]])
# census_data_split = np.zeros((len(census_data),2,2))
# for i in range(len(census_data)):
#     for v in range(2):
#         for w in range(2):
#             census_data_split[i,v,w] = census_data[i]*(prevalence_WorA_ethnicity[i] if v==1 else 1-prevalence_WorA_ethnicity[i])*(prevalence_high_contact[i,v] if w==1 else 1-prevalence_high_contact[i,v])
# homophily_attribute1 = 1
# multipliers_attribute2 = [1,2]
# m = ContactMatrix2(empirical_contact_matrix,census_data,census_data_split,homophily_attribute1,multipliers_attribute2)
# CC = m.solve(homophily_attribute1)
# (C_out,N_out,indices_out) = m.delete_empty_populations(CC)
# #plt.imshow(np.log(C_out))
# plt.imshow(np.log(C_out-1e-5))


# prevalence_comorbidities = np.array([0,0.3,0.7,0.8]) #not correct values
# prevalence_comorbidities = np.array([prevalence_comorbidities[i[0]] for i in indices_out])
# m2 = ContactMatrix(C_out,N_out,prevalence_comorbidities)
# CC2 = m2.solve(0)

# (C_out,N_out,indices_out) = m2.delete_empty_populations(CC2)
# print('reciprocal:',m.issymmetric(C_out,N_out))



def get_contact_matrix_ISMART(empirical_contact_matrix,census_data,prevalence_WorA_ethnicity,prevalence_high_contact,homophily_ethnicity,multipler_highcontact_jobs,prevalence_comorbidities):
    census_data_split = np.zeros((len(census_data),2,2))
    for i in range(len(census_data)):
        for v in range(2):
            for w in range(2):
                census_data_split[i,v,w] = census_data[i]*(prevalence_WorA_ethnicity[i] if v==1 else 1-prevalence_WorA_ethnicity[i])*(prevalence_high_contact[i,v] if w==1 else 1-prevalence_high_contact[i,v])
    homophily_attribute1 = homophily_ethnicity
    multipliers_attribute2 = [1,multipler_highcontact_jobs]
    m = ContactMatrix2(empirical_contact_matrix,census_data,census_data_split,homophily_attribute1,multipliers_attribute2)
    CC = m.solve(homophily_attribute1)
    (C_out,N_out,indices_out) = m.delete_empty_populations(CC)
    
    
    prevalence_comorbidities = np.array([prevalence_comorbidities[i[0]] for i in indices_out])
    m2 = ContactMatrix(C_out,N_out,prevalence_comorbidities)
    CC2 = m2.solve(0)
    (C_out,N_out,indices_out2) = m2.delete_empty_populations(CC2)
    
    indices = []
    for index_old,j in indices_out2:
        indices.append(np.append(indices_out[index_old],j))
    
    return (C_out,N_out,np.array(indices))

def get_contact_matrix_ISMART_nocomorbidities(empirical_contact_matrix,census_data,prevalence_WorA_ethnicity,prevalence_high_contact,homophily_ethnicity,multipler_highcontact_jobs):
    census_data_split = np.zeros((len(census_data),2,2))
    for i in range(len(census_data)):
        for v in range(2):
            for w in range(2):
                census_data_split[i,v,w] = census_data[i]*(prevalence_WorA_ethnicity[i] if v==1 else 1-prevalence_WorA_ethnicity[i])*(prevalence_high_contact[i,v] if w==1 else 1-prevalence_high_contact[i,v])
    homophily_attribute1 = homophily_ethnicity
    multipliers_attribute2 = [1,multipler_highcontact_jobs]
    m = ContactMatrix2(empirical_contact_matrix,census_data,census_data_split,homophily_attribute1,multipliers_attribute2)
    CC = m.solve(homophily_attribute1)
    (C_out,N_out,indices_out) = m.delete_empty_populations(CC)
    return (C_out,N_out,np.array(indices_out))


# homophily_ethnicity = 0.8
# multipler_highcontact_jobs = 10
# prevalence_high_contact = np.array([[0,0],[0.5,0.25],[0,0],[0,0]])

# prevalence_comorbidities = np.array([0,0.3,0.7,0.8]) #not correct values
# (C_out,N_out,indices_out) = get_contact_matrix_ISMART(empirical_contact_matrix,census_data,prevalence_WorA_ethnicity,prevalence_high_contact,homophily_ethnicity,multipler_highcontact_jobs,prevalence_comorbidities)
# plt.imshow(np.log(C_out))
# plot_contact_matrix(C_out,N_out,indices_out,[],'ssdas.pdf')


# #analysis
# CC_specific_homophily_values = m.get_specific_homophily_values(CC)



# names_attributes = [['0-15','16-64','65-74','75+'],['Not White and not Asian','White or Asian'],['low-contact job','high-contact job']]










# plot_4x4_contact_matrix(C_old,'nonsymmetric_empirical_contact_matrix.pdf')


# for homophily in [0,0.5,1]:

#     M = BuildContactMatrix(C_old,people_per_age,p_attribute1,homophily_attribute1,norm=norm)
#     x = M.solve(homophily)
#     contact_matrix = M.asmatrix(x)
    
#     final_contact_matrix,final_N8 = M.delete_empty_populations(contact_matrix,M.N)
#     plot_8x8_contact_matrix(final_contact_matrix,'final_8x8_contactmatrix_homophily%i.pdf' % int(round(homophily*1000)))
    
#     attribute1_TRUE,attribute1_FALSE = M.get_attribute1_specific_age_age_contact_matrices(x)
#     plot_4x4_contact_matrix(attribute1_TRUE,'white_ageage_contact_matrix_homophily%i.pdf' % int(round(homophily*1000)))
#     plot_4x4_contact_matrix(attribute1_FALSE,'black_ageage_contact_matrix_homophily%i.pdf' % int(round(homophily*1000)))    
        
    
    
#     M2 = BuildContactMatrix( contact_matrix,M.N,p_attribute2,0,norm=norm,contact_multiplier_groupTRUE=2)
#     xx = M2.solve(0,tol=1e-4)
#     contact_matrix2 = M2.asmatrix(xx)
    
#     M3 = BuildContactMatrix( contact_matrix2,M2.N,p_attribute3,0,norm=norm)
#     xx = M3.solve(0,tol=1e-4)
#     contact_matrix3 = M3.asmatrix(xx)
    
#     final_contact_matrix,final_N18 = M3.delete_empty_populations(contact_matrix3,M3.N)
    
#     plot_contact_matrix(final_contact_matrix,'final_18x18_contactmatrix_homophily%i.pdf' % int(round(homophily*1000)))


  

# import numpy as np  
# C_old = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
# people_per_age = np.array([ 60570126, 213610414,  31483433,  22574830])
# p_attribute1 = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
# homophily_attribute1 = 0
# p_attribute2 = np.array([[1,1],[0.65,0.65],[1-0.79602454,1-0.79602454],[1-0.82338084,1-0.82338084]])
# p_attribute3 = np.zeros((4,2,2))
# p_attribute3[1] = np.array([[0.25,0.25],[0.4,0.4]])

# C_old = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
# N_old = np.array([ 60570126, 213610414,  31483433,  22574830])
# p_attribute1 = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
# homophily_attribute1 = 0
# p_attribute2 = np.zeros((4,2))
# p_attribute2[1,0] = 0.25
# p_attribute2[1,1] = 0.5
# p_attribute3 = np.ones((4,2,2))
# p_attribute3[1] = 0.71
# p_attribute3[2] = 0.282000012
# p_attribute3[3] = 0.21699997



# for i in range(4):
#     print(people_per_age[i]*p_attribute1[i])




#Example data
values_p = np.array([0.0001,0.001,0.01,0.1,0.2,0.3,0.4])
values_p = np.r_[values_p,0.5,1-values_p]
values_p.sort()


POSITIVE = np.zeros((len(values_p),len(values_p)))
for ii,p1 in enumerate(values_p):
    for jj,p2 in enumerate(values_p):
        C = np.array([[9,12],[4,15]])
        N = np.array([100,300])
        prevalence_attribute = np.array([p1,p2])
        m = ContactMatrix(C,N,prevalence_attribute)
        #print(m.issymmetric(C,N))
        S=m.symmetrize_Funk2020(C,N)
        #print(m.issymmetric(S,N))
        CC_small = m.get_contact_matrix_when_homophily0()
        (LHS,RHS) = m.get_equality_contraints_matrix(0)
        contact_matrix = m.solve(0.99)
        #print(p2,np.min(contact_matrix))
        #print(m.get_specific_homophily_values(contact_matrix))
        #print(m.issymmetric(contact_matrix,m.N))
        POSITIVE[ii,jj] = np.min(contact_matrix) > 0
        

HIGHEST_POSSIBLE_HOMOPHILY = np.zeros((len(values_p),len(values_p)))
for ii,p1 in enumerate(values_p):
    for jj,p2 in enumerate(values_p):
        C = np.array([[9,12],[4,15]])
        N = np.array([100,300])
        prevalence_attribute = np.array([p1,p2])
        m = ContactMatrix(C,N,prevalence_attribute)
        #print(m.issymmetric(C,N))
        S=m.symmetrize_Funk2020(C,N)
        #print(m.issymmetric(S,N))
        CC_small = m.get_contact_matrix_when_homophily0()
        (LHS,RHS) = m.get_equality_contraints_matrix(0)
        h_high=1
        h_low=0
        h_med = 1
        for i in range(20):
            m.get_null_space(h_med)
            CC_small_nnls,error = opti.nnls(m.design_matrix,m.RHS)
            #contact_matrix = m.solve(h_med)
            if error<1e-10:#np.min(contact_matrix) > 0:
                if h_med==1:
                    break
                h_low = h_med
            else:
                h_high = h_med
            h_med = (h_low+h_high)/2
        HIGHEST_POSSIBLE_HOMOPHILY[ii,jj] = h_med
        
f,ax = plt.subplots(figsize=(3,3))
im = ax.imshow((HIGHEST_POSSIBLE_HOMOPHILY.T),origin='lower',vmin=0,vmax=1,cmap=cm.Blues)
ax.set_xticks(np.arange(len(values_p)))
ax.set_xticklabels(list(map(str,values_p)),rotation=90)
ax.set_yticks(np.arange(len(values_p)))
ax.set_yticklabels(list(map(str,values_p)),rotation=0)
ax.set_xlabel(r'$P_1$')
ax.set_ylabel(r'$P_2$')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = f.colorbar(im,cax=cax)
cbar.set_label('highest possible homophily')  
plt.savefig(infix+'possible_homophilies_2_agegroups.pdf',bbox_inches = "tight")



C = np.array([[9,12],[4,15]])
N = np.array([100,300])
nsim=1000
dims = np.zeros(nsim)
errors = np.zeros(nsim)
ps = np.zeros((nsim,2))
for i in range(nsim):
    p1,p2 = np.random.random(2)
    h=1

    prevalence_attribute = np.array([p1,p2])
    m = ContactMatrix(C,N,prevalence_attribute)
    #print(m.issymmetric(C,N))
    #S=m.symmetrize_Funk2020(C,N)
    #print(m.issymmetric(S,N))
    m.get_null_space(h)
    CC_small_nnls,error = opti.nnls(m.design_matrix,m.RHS)
    
    dims[i]=(m.null.shape[1])
    errors[i]=(error)
    ps[i]=([p1,p2])
    
threshold = 1e-10
f,ax = plt.subplots(figsize=(3,3))
ax.plot(ps[errors>threshold,0],ps[errors>threshold,1],'rx')
ax.plot(ps[errors<threshold,0],ps[errors<threshold,1],'bx')

        
        
        

#C = np.array(pd.read_csv('prem2020_US_contact_all.csv'))#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])






        
        
C = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
N = np.array([ 60570126, 213610414,  31483433,  22574830])
nsim=100000
dims = np.zeros(nsim)
errors = np.zeros(nsim)
ps = np.zeros((nsim,4))
for i in range(nsim):
    prevalence_attribute = np.random.random(4)
    h=1

    m = ContactMatrix(C,N,prevalence_attribute)
    #print(m.issymmetric(C,N))
    #S=m.symmetrize_Funk2020(C,N)
    #print(m.issymmetric(S,N))
    m.get_null_space(h)
    CC_small_nnls,error = opti.nnls(m.design_matrix,m.RHS)
    
    dims[i]=(m.null.shape[1])
    errors[i]=(error)
    ps[i]=prevalence_attribute 
        
threshold = 1e-6
bins = 10
std_hist_failed,xs = np.histogram(np.std(ps[errors>threshold],1),bins=bins,range=(0,0.5))
std_hist_worked = np.histogram(np.std(ps[errors<threshold],1),bins=bins,range=(0,0.5))[0]
f,ax = plt.subplots(figsize=(3,3))
ax.bar((xs[1:]+xs[:-1])/2,std_hist_failed/nsim,color='r',width=0.9*0.5/bins) 
ax.bar((xs[1:]+xs[:-1])/2,std_hist_worked/nsim,bottom=std_hist_failed/nsim,color='b',width=0.9*0.5/bins) 
ax.set_xlabel('std(P)')
ax.set_ylabel('proportion')
plt.savefig(infix+'4x4_us_matrix_homophily1_possible_std_nsim%i.pdf' % nsim,bbox_inches = "tight")

maxdiff_hist_failed,xs = np.histogram(np.max(ps[errors>threshold],1)-np.min(ps[errors>threshold],1),bins=bins,range=(0,1))
maxdiff_hist_worked = np.histogram(np.max(ps[errors<threshold],1)-np.min(ps[errors<threshold],1),bins=bins,range=(0,1))[0]
f,ax = plt.subplots(figsize=(3,3))
ax.bar((xs[1:]+xs[:-1])/2,maxdiff_hist_failed/nsim,color='r',width=0.9*1/bins) 
ax.bar((xs[1:]+xs[:-1])/2,maxdiff_hist_worked/nsim,bottom=maxdiff_hist_failed/nsim,color='b',width=0.9*1/bins) 
ax.set_xlim([0,1])
ax.set_xlabel('max(P) - min(P)')
ax.set_ylabel('proportion')
plt.savefig(infix+'4x4_us_matrix_homophily1_possible_maxdiff_nsim%i.pdf' % nsim,bbox_inches = "tight")

        
bins=50
for i in range(4):
    Pi_hist_failed,xs = np.histogram(ps[errors>threshold,i],bins=bins,range=(0,1))
    Pi_hist_worked = np.histogram(ps[errors<threshold,i],bins=bins,range=(0,1))[0]
    f,ax = plt.subplots(figsize=(2.2,2.2))
    #ax.bar((xs[1:]+xs[:-1])/2,Pi_hist_failed/nsim,color='r',width=0.9*1/bins) 
    #ax.bar((xs[1:]+xs[:-1])/2,Pi_hist_worked/nsim,bottom=Pi_hist_failed/nsim,color='b',width=0.9*1/bins) 
    ax.bar((xs[1:]+xs[:-1])/2,Pi_hist_failed/(Pi_hist_worked+Pi_hist_failed),color='k',width=0.9*1/bins) 
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(r'$P_%i$' % (i+1))
    ax.set_ylabel('proportion')
    plt.savefig(infix+'4x4_us_matrix_homophily1_possible_P%i_nsim%i.pdf' % (i+1,nsim),bbox_inches = "tight")

bins=50
f,ax = plt.subplots(nrows=1,ncols=4,figsize=(5.2,1.2),sharey=True)
for i in range(4):
    Pi_hist_failed,xs = np.histogram(ps[errors>threshold,i],bins=bins,range=(0,1))
    Pi_hist_worked = np.histogram(ps[errors<threshold,i],bins=bins,range=(0,1))[0]
    #ax.bar((xs[1:]+xs[:-1])/2,Pi_hist_failed/nsim,color='r',width=0.9*1/bins) 
    #ax.bar((xs[1:]+xs[:-1])/2,Pi_hist_worked/nsim,bottom=Pi_hist_failed/nsim,color='b',width=0.9*1/bins) 
    ax[i].bar((xs[1:]+xs[:-1])/2,Pi_hist_failed/(Pi_hist_worked+Pi_hist_failed),color='k',width=0.9*1/bins) 
    ax[i].set_xlim([0,1])
    ax[i].set_ylim([0,1])
    ax[i].set_xlabel(r'$P_%i$' % (i+1))
    ax[i].set_xticks([0,0.5,1])
    ax[i].set_xticklabels(list(map(str,[0,0.5,1])))
    ax[i].set_yticks([0,0.5,1])
    ax[i].set_yticklabels(list(map(str,[0,0.5,1])))
    if i==0:
        ax[i].set_ylabel('proportion $h_{\max} < 1$')
plt.subplots_adjust(wspace = .09,bottom=0.02,top=0.99,right=0.99)
plt.savefig(infix+'4x4_us_matrix_homophily1_possible_Pall_nsim%i.pdf' % (nsim),bbox_inches = "tight")


bins = 40
FILTER=0
from scipy.signal import savgol_filter
f,ax = plt.subplots(figsize=(3,3))
for i in range(4):
    Pi_hist_failed,xs = np.histogram(ps[errors>threshold,i],bins=bins,range=(0,1))
    Pi_hist_worked = np.histogram(ps[errors<threshold,i],bins=bins,range=(0,1))[0]
    ratio = Pi_hist_failed/(Pi_hist_failed+Pi_hist_worked)
    ax.plot((xs[1:]+xs[:-1])/2,savgol_filter(ratio,3,1) if FILTER else ratio ,'-',label=r'$P_%i$' % (i+1))
ax.legend(loc='best',frameon=0)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
C = np.array([[9,12],[4,15]])
N = np.array([100,300])
p1,p2 = 0.5,0.8
prevalence_attribute = np.array([p1,p2])
m = ContactMatrix(C,N,prevalence_attribute)
CC_small = m.get_contact_matrix_when_homophily0()

h = 0.5
m.get_null_space(h)
CC_small_nnls,error = opti.nnls(m.design_matrix,m.RHS)
CC_small_nnls = np.reshape(CC_small_nnls,(2,2,2,2))
print(error)
print(np.min(CC_small_nnls),np.round(CC_small_nnls,3))






#find optimum w.r.t. objective 1







import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

def SIR_mixing(x,t,beta,r,contact_square_matrix):
    n = len(contact_square_matrix)
    S = x[:n]
    I = x[n:2*n]
    dS = np.zeros(n)
    dI = np.zeros(n)
    dR = np.zeros(n)
    for i in range(n):
        force_of_infection = sum([I[j] * contact_square_matrix[i,j] for j in range(n)])
        dS[i] = - beta*S[i]*force_of_infection
        dI[i] = beta*S[i]*force_of_infection - r*I[i]
        dR[i] = r*I[i]
    return np.r_[dS,dI,dR]


T=100
dt=0.1
ts = np.linspace(0, T, int(T/dt)+1)
beta = 0.5e-1
r = 1e-1

C = np.array([[9,12],[4,15]])
N = np.array([100,300])
p1,p2 = 0.5,0.8
prevalence_attribute = np.array([p1,p2])

m = ContactMatrix(C,N,prevalence_attribute)
CC_small = m.get_contact_matrix_when_homophily0()

for h in [0,0.5,0.8,0.95,0.98,1]:
    C_opt = m.solve(h)
    
    n = np.prod(C_opt.shape[:2])
    contact_square_matrix = C_opt.reshape(n,n)
    x0_dummy = m.N.reshape(n)
    x0_dummy = x0_dummy/sum(x0_dummy)
    
    #seed epidemic
    proportion_initially_infected = [1e-2,0,0,0]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[i] = (1-proportion_initially_infected[i])*x0_dummy[i]
        x0[n+i] = proportion_initially_infected[i]*x0_dummy[i]
        
    
    sol = integrate.odeint(SIR_mixing, x0, ts, args=(beta, r, contact_square_matrix))
    
    
    proportion_infected = np.divide(sol[:,n:2*n],x0_dummy)
    colors = ['b','orange']
    lss = ['-','--']
    ages = ['young','old']
    labels = ['False','True']
    f,ax = plt.subplots(figsize=(2.5,2.2))
    for i in range(n):
        ax.plot(ts,proportion_infected[:,i],color=colors[i%2],ls=lss[i//2],label=ages[i//2]+' & '+labels[i%2])
    ax.set_xlabel('time')
    ax.set_ylabel('proportion infected')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.legend(loc='best',frameon=False)
    ax.set_ylim([-0.03,0.5])
    plt.savefig('example_h%f.pdf' % (h),bbox_inches = "tight")





def SIR_mixing_demo(x,t,beta,r,mu,contact_square_matrix):
    n = len(contact_square_matrix)
    S = x[:n]
    I = x[n:2*n]
    R = x[2*n:3*n]
    dS = np.zeros(n)
    dI = np.zeros(n)
    dR = np.zeros(n)
    for i in range(n):
        force_of_infection = sum([I[j] * contact_square_matrix[i,j] for j in range(n)])
        dS[i] = mu - beta*S[i]*force_of_infection  - mu*S[i]
        dI[i] = beta*S[i]*force_of_infection - r*I[i] - mu*I[i]
        dR[i] = r*I[i] - mu*R[i]
    return np.r_[dS,dI,dR]




#real data
T=2500
dt=0.1
ts = np.linspace(0, T, int(T/dt)+1)
beta = 0.3e-1
r = 1e-1
mu = 1e-4

C = np.array(pd.read_csv('interactionmatrix/age_interactions.csv'))[:,1:]#np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
N = np.array([ 60570126, 213610414,  31483433,  22574830])
prevalence_attribute = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])

m = ContactMatrix(C,N,prevalence_attribute)
CC_small = m.get_contact_matrix_when_homophily0()

#for h in [0,0.5,0.8,0.95,0.98,1]:
for h in [0]:
    C_opt = m.solve(h)
    
    n = np.prod(C_opt.shape[:2])
    contact_square_matrix = C_opt.reshape(n,n)
    x0_dummy = m.N.reshape(n)
    x0_dummy = x0_dummy/sum(x0_dummy)
    
    #seed epidemic
    proportion_initially_infected = [1e-2,0,0,0]
    proportion_initially_infected = [0,1e-2,0,0,0,0,0,0]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[i] = (1-proportion_initially_infected[i])*x0_dummy[i]
        x0[n+i] = proportion_initially_infected[i]*x0_dummy[i]
        
    
    sol = integrate.odeint(SIR_mixing_demo, x0, ts, args=(beta, r, mu, contact_square_matrix))
    
    
    proportion_infected = np.divide(sol[:,n:2*n],x0_dummy)
    colors = ['b','orange','k','g']
    lss = ['-','--']
    ages = ['0-14','15-64','65-74','75+']
    labels = ['False','True']
    f,ax = plt.subplots(figsize=(4.5,3.2))
    for i in range(n):
        ax.plot(ts,proportion_infected[:,i],ls=lss[i%2],color=colors[i//2],label=ages[i//2]+' & '+labels[i%2])
    ax.set_xlabel('time')
    ax.set_ylabel('proportion infected')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',frameon=False)
    ax.set_ylim([-0.03,0.4])
    plt.savefig('example_h%f.pdf' % (h),bbox_inches = "tight")











