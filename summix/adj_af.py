### Adjusting Allele Frequences
### Main feature: adjAF, an efficient method for adjusting allele frequencies
### Authors: Jordan R. Hall, Kaichao Chang; Supervised by Dr. Audrey Hendricks
### Original research of Katie Marker. This code is entirely based on her original work.
### Mixtures Research Group, Dr. Audrey E. Hendricks, Univ. of Colorado Denver, 2020.

### Import needed packages
import numpy as np
import pandas as pd
from summix.summix import data_processor

### adjAF: (ref, obs, pi_target, file_name, file_format, pi_hat) -> new_DF

### A data-processing function that takes 8 inputs:

 ## 1. ref: A list of strings containing which columns the user wants to use for reference allele frequencies. 
 ##    Pass the name of each column as a string. 
 ##    So for example, if the desired reference ancestries are called "ref_eur_1000G" and "ref_afr_1000G", then use
 ##    ref=['ref_eur_1000G','ref_afr_1000G'].
 
 ## 2. obs: Which column to use for observed allele freq's. Pass the name of this column, as a string.
 ##         So for example, if the desired observed ancestry is stored in a column called "gnomAD_afr", then obs='gnomAD_afr'. 
 
 ## 3. pi_target: Updated proportions of each reference ancestry within some true population.
 ##             Values should be provided in the same order as references are provided in!!!

 ## 4. file_name: A user-input genetic data "file". 
 ##               Must be a .txt or a .csv file (with the .txt or .csv as the last four characters of the actual file name).
 ##               See data formatting standards for more info about required rows/columns.

 ## 5. file_format: The "file_format" of file, as a string. 
 ##                 Default is 'tab' which is short for tab-delimited text files. Can also choose 'csv' for CSV files.
 
 ## 6. pi_hat: The estimated proportions of each reference ancestry within the observered or modeled population.
 ##            Values should be provided in the same order as references are provided in!!!
 ##            pi_hat is an optional argument if the user provided allele freq's for all references, and will be solved via summix

### and returns 1 output:

 ## 1. new_DF: Genetic data in an input array "A" size Nxk containing N SNPs (these are the rows), and k reference ancestries (these are the columns);

def adjAF(ref, obs, pi_target, file_name, file_format, pi_hat=None):

    k = len(ref) #grab number of provided reference ancestries
    
    # Check that user has provided pi_target and pi_hat of correct length
    if np.shape(pi_target)[0] != k:
        print('Please ensure pi_target has as many entries as there are references provided.')
        return
        
    if np.shape(pi_hat)[0] != k and pi_hat != None:
        print('Please ensure pi_hat has as many entries as there are references provided.')
        return
        
    pi_target = np.copy(pi_target)/np.sum(pi_target) # normalize pi_target

    # Reads data file in using pandas, depending on file_format
    if (file_format=='csv') == True:
        D = pd.read_csv(file_name)
    else:
        D = pd.read_csv(file_name, sep='\t')
    
    names = D.columns # collect list of column names in provided data frame
    
    # Now we count how many references are actually in the data frame, because sometimes the user may be missing 1 
    # (which is OK here, we just need to know which one is missing...)
    ref_count=0
    missing_ref_index = k-1 # default
    for i in range(0,np.shape(ref)[0]):
        if (ref[i] in names) == True:
            ref_count=ref_count+1
        else:
            missing_ref_index = i
            print('Note: There is no allele frequency data provided for the',ref[missing_ref_index],' ancestry. One missing ancestry is permitted in this formulation. \n \n \n')
    
   # Confusing/not needed, but not quite ready to delete yet...
   # if ref_count == k:
        #print('Note: Because all allele frequencies were provided for all ancestries, the',ref[missing_ref_index],'ancestry is not used in the formulation.\n \n \n')
            
    if pi_hat is None and ref_count==k:
        answer_obj = HA_script.SUMMIX(ref=ref,obs=obs, file=file, k=k, x_guess=None, file_format=file_format)
        pi_hat = answer_obj[0] # Defines pi_final as the solution vector
        print('Because pi_hat was unspecified, pi_hat has been estimated using the HA script with your specified inputs. \n \n \n', 'The resulting pi_hat is shown in the full HA printout above. \n \n \n')

    elif pi_hat is None and ref_count<k:
        print('Because one of the reference ancestries cannot be found in the provided data frame, we cannot use summix to provide an estimate for pi_target. \n \n \n', ' Please provide an estimate for pi_target. \n \n \n')
        return

    pi_hat = np.copy(pi_hat)/np.sum(pi_hat) # normalize pi_hat
    
    # Form needed constant from adjAF formula (see paper)
    C = pi_target[missing_ref_index]/pi_hat[missing_ref_index]

    # Remove the name for the missing reference from reference list. Default is the last one, if none are missing.
    ref.pop(missing_ref_index)
    
    D_pr_ref = D[ref] # grabs references for printing
    D_pr_obs = D[obs] # grabs observed for printing
    
    print('By default we print out the first 5 rows of the user-provided reference ancestries \n \n that will be used in the calculation. Please check column names for correctness: \n \n \n',D_pr_ref.head(5), '\n \n \n We also print out the first 5 rows of the user-provided observed ancestry \n \n  to be used in the calculation. Please check column name for correctness: \n \n \n',D_pr_obs.head(5), '\n \n \n')

    
    # Use the data_processor to take the info we need out of the data frame D
    data_array = data_processor(file_name, file_format, k-1, ref, obs)
    ref_AFs = data_array[0]
    observed_AFs = data_array[1]
        
    # Instantiate first term in summation
    temp = C*observed_AFs
    
    # Perform summation except at py_adj_index (which has already been excluded from ref_AFs
    for i in range(0,k-1):
        temp = np.copy(temp) - C*pi_hat[i]*ref_AFs[:,i:(i+1)] + pi_target[i]*ref_AFs[:,i:(i+1)]

    # Routine for rounding estimates
    for i in range(0,np.shape(temp)[0]):
        if temp[i] <0:
            temp[i]=0.0
        elif temp[i]>1:
            temp[i]=1.0
        else:
            temp = np.copy(temp)

    # This is our answer, a new vector/column of ancestry-adjusted allele frequencies
    adj_AF = temp
    
    # Merge the adj_AF into the original data (Kaichao's contribution! Thanks Kaichao!) as an additional column called adjusted_AF
    updateAF = pd.DataFrame(data=adj_AF, columns=['adjusted_AF'])
    new_DF = pd.concat([D, updateAF], axis=1)
    new_DF['adjusted_AF'] = new_DF['adjusted_AF'].astype(str)    

    print('Adjustment complete! \n \n', 'A data frame called new_DF has been created for the user. \n \n', 'new_DF contains the original, user-provided data frame, \n \n', 'concatenated with the adjusted allele frequencies (appended as the last column).')

    return new_DF
