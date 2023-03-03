# HomophilicContactMatrices

This program implements a new method described in this paper: https://www.aimspress.com/article/doi/10.3934/mbe.2023154?viewType=HTML.

It projects age-structured social contact matrices to populations split not only by age but by further Boolean attributes. For each Boolean attribute, its prevalence across the population is known.
For attribute1, the population may also exhibit positive homophily, h in [0,1], such that there are more contacts among people with the same attribute value. 
The number of interactions between individuals with different attribute1 values is (1-h) times the expected number (i.e. in the case h==0), where attribute 1 has no impact on social contact levels. 
In the other extreme case (h==1), there exists complete segregation: only individuals with the same attribute1 value interact.

The base class is ContactMatrix, which takes as input:
1. an empirical mxm contact matrix (not necesarily "symmetric"), e.g. stratified only by age
2. population sizes for each of the m groups (absolute numbers of proportion), i.e. a vector of length m
3. the prevalence of the newly added binary attribute in each of the m groups, i.e. a vector of length m with values in [0,1].
    
To create an expanded contact matrix stratified by the additional binary attribute with homophily=0.9, run:
```python
C = np.array([[9,12],[4,15]])
N = np.array([100,300])
prevalence_attribute = np.array([0.5,0.8])    
CM = ContactMatrix(C,N,prevalence_attribute)
expanded_homophilic_contact_matrix = CM.solve(homophily=0.9)  
```

This yields a mx2xmx2 matrix. To transform this back into a usual square matrix, which we can plot, run:
```python
(expanded_homophilic_contact_matrix_square,expanded_N,expanded_indices) = m.delete_empty_populations(expanded_homophilic_contact_matrix)
names_attributes = [['G1','G2'],['True','False']]
plot_contact_matrix_specific(expanded_homophilic_contact_matrix_square,expanded_N,expanded_indices,names_attributes,'test.pdf')
```
    
The second class is ContactMatrix2, which should be used if, in addition to the added binary homophilic attribute1, 
there is a second binary attribute2, which splits the population into high- and low-contact individuals. The inputs are:
1. as before, an empirical mxm contact matrix (not necesarily "symmetric"), e.g. stratified only by age
2. the population sizes for each of the m x 2 (attribute 1) x 2 (attribute2) groups (absolute numbers of proportion), i.e. an array of size m x 2 x 2
3. multipliers_attribute2, an array of size 2, describing the relative contact level of people with attribute2 == True vs people with attribute2 == False. One may always use [1, x] where x>0. 
    
Given an original 4x4 age-age matrix of the U.S., to create an expanded contact matrix with homophily=0.9 w.r.t an added binary attribute 'ethnicity'
including also high-contact and low-contact employees, run:
```python
empirical_contact_matrix = np.array([[7.48,5.05,0.18,0.04],[1.96,12.12,0.21,0.04],[0.93,3.75,1.14,0.15],[0.91,2.70,0.49,0.40]])
census_data = np.array([ 60570126, 213610414,  31483433,  22574830])
prevalence_WorA_ethnicity = np.array([0.5481808309264538, 0.652413561634687, 0.7960244996154009, 0.8233809069658553])
prevalence_high_contact = np.array([[0,0],[0.45,0.15],[0,0],[0,0]])
homophily_attribute1 = 0.9
census_data_split = np.zeros((len(census_data),2,2))
for i in range(len(census_data)):
    for v in range(2):
        for w in range(2):
            census_data_split[i,v,w] = census_data[i]*(prevalence_WorA_ethnicity[i] if v==1 else 1-prevalence_WorA_ethnicity[i])*(prevalence_high_contact[i,v] if w==1 else 1-prevalence_high_contact[i,v])
multipliers_attribute2 = [1,3]
CM = ContactMatrix2(empirical_contact_matrix,census_data_split,multipliers_attribute2)
expanded_homophilic_contact_matrix = CM.solve(homophily = homophily_attribute1)
```

This yields a mx2x2 x mx2x2 matrix. To transform this back into a usual square matrix, which we can plot, run:
```python
names_attributes = [['0-15','16-64','65-74','75+'],['POC','WA'],['LC','HC']]
(expanded_homophilic_contact_matrix_square,expanded_N,expanded_indices) = CM.delete_empty_populations(expanded_homophilic_contact_matrix)
plot_contact_matrix_specific(expanded_homophilic_contact_matrix_square,expanded_N,expanded_indices,names_attributes,'JTB_study_homotphily%i.pdf' % int(100*homophily_attribute1))
```
