import msprime
import numpy as np
import pandas as pd
from array import array
import random
from sklearn.decomposition import PCA


#sample=60000
#will generate 30k patients and number of snpa determined by the msprime
#300k-


filenumber = input("Please enter the number of files to be generated: ")
print("You entered: " + filenumber)

def generate_and_save_matrix(filenumber):
#sample=4096
#num_snps = 1024000-2

    sample = input("Please enter number of samples (choose it to be multiple of tile size): ")
    print("You entered: " + sample)

    lens = input("Please enter the length of the simulated region if ( 300K SNPs -> enter 1e7, for 600K SNPs-> enter 2e7, and for 1M SNPs -> enter 3e7): ")
    print("You entered: " + lens)

    for i in range(filenumber):
        lens =  float(lens) #2e7 #3e7
        ts = msprime.simulate(
            sample_size=2*int(sample),
            Ne=1e4,
            length=lens,
            recombination_rate=2e-9,
            mutation_rate=1e-7,
            random_seed = 4533
        )

        print('Total number of SNPs {}'.format(ts.num_sites))
        if ts.num_sites < 30000:
            num_snps = 204800-2
        elif (ts.num_sites < 40000 and ts.num_sites > 30000):
            num_snps = 30720-2
        elif (ts.num_sites < 60000 and ts.num_sites > 40000):
            num_snps = 51200-2
        elif (ts.num_sites < 80000 and ts.num_sites > 60000):
            num_snps = 61440-2
        elif (ts.num_sites < 100000 and ts.num_sites > 80000):
            num_snps = 81920-2
        elif (ts.num_sites < 120000 and ts.num_sites > 100000):
            num_snps = 102400-2
        elif (ts.num_sites < 140000 and ts.num_sites > 120000):
            num_snps = 122880-2
        elif (ts.num_sites < 160000 and ts.num_sites > 140000):
            num_snps = 143360-2
        elif (ts.num_sites < 180000 and ts.num_sites > 160000):
            num_snps = 163840-2
        elif (ts.num_sites < 200000 and ts.num_sites > 180000):
            num_snps = 184320-2
        elif (ts.num_sites < 220000 and ts.num_sites > 200000):
            num_snps = 204800-2
        elif (ts.num_sites < 240000 and ts.num_sites > 220000):
            num_snps = 225280-2
        elif (ts.num_sites < 260000 and ts.num_sites > 240000):
            num_snps = 245760-2
        elif (ts.num_sites < 280000 and ts.num_sites > 260000):
            num_snps = 266240-2
        elif (ts.num_sites < 300000 and ts.num_sites > 280000):
            num_snps = 286720-2
        elif (ts.num_sites < 500000 and ts.num_sites > 300000): #1e7
            num_snps = 307200-2
        elif (ts.num_sites < 800000 and ts.num_sites > 500000): #2e7
            num_snps = 614400-2
        elif (ts.num_sites < 1200000 and ts.num_sites > 700000): #3e7
            num_snps = 1024000-2
        elif (ts.num_sites < 3000000 and ts.num_sites > 1200000): #4e7
            num_snps = 2048000-2
        elif (ts.num_sites < 5000000 and ts.num_sites > 3000000):#9e7
            num_snps = 3072000-2
        elif (ts.num_sites < 8000000 and ts.num_sites > 5000000): #
            num_snps = 6144000-2
        elif (ts.num_sites < 12000000 and ts.num_sites > 8000000):
            num_snps = 10240000-2
        elif (ts.num_sites < 16000000 and ts.num_sites > 12000000):
            num_snps = 12288000-2

        print("Number of trancated SNPs: " + str(num_snps))
        num_snps = input("Number of SNPs: ")
        print("You entered: " + num_snps)
        num_snps=int(num_snps)-2

        #with open("simulation8.vcf", "w") as vcf_file:
        #    ts.write_vcf(vcf_file, ploidy=2)

        # Generate the genotype matrix with 0, 1, 2 values by summing the two alleles
        tsg = ts.genotype_matrix()
        tsg2 = np.add.reduceat(tsg, range(0, tsg.shape[1], 2), axis=1)
        constant_alleles = np.isclose(tsg2.std(axis=1), 0)

        tsg2 = np.delete(tsg2, constant_alleles, axis=0)
        tsg2 = tsg2[:num_snps,:]
        con = tsg2.transpose()
        #print('Genotype matrix (number of SNPs = {}, number of samples = {})'.format(tsg2.shape[0], tsg2.shape[1]))
        #tsg2 = tsg2.transpose()
        #print('Genotype matrix (number of samples = {}, number of SNPs = {})'.format(tsg2.shape[0], tsg2.shape[1]))




        genetic_variance = .7 # proportion of phenotypic variance explained by the genotypic variation
        environmental_variance = .15 # proportion of phenotypic variance explained by the environment
        relatedness_variance = .15 # proportion of phenotypic variance explained by population stratification


        row_means = tsg2.mean(axis=1)
        row_stds = tsg2.std(axis=1)

        tsg2_norm = (tsg2 - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]



        # We consider the setup where 10% of the SNPs are causal
        num_causal_SNPs = tsg2.shape[0] // 10
        pos_causal_SNPs = np.array(random.sample(range(tsg2.shape[0]), num_causal_SNPs))
        pos_causal_SNPs = np.sort(pos_causal_SNPs)

        weight_causal_SNPs = np.zeros((1, tsg2_norm.shape[0]))
        weight_causal_SNPs[:,pos_causal_SNPs] = np.random.normal(scale=np.sqrt(genetic_variance / num_causal_SNPs),
                                                                 size=num_causal_SNPs)

        environment_component = np.random.normal(scale=np.sqrt(environmental_variance), size=(1, tsg2_norm.shape[1]))
        relatedness_component = PCA(n_components=1).fit(tsg2_norm).components_
        relatedness_component = relatedness_component / relatedness_component.std(axis=1) * np.sqrt(relatedness_variance)


        # Continous outcome
        phenotype_continuous = weight_causal_SNPs @ tsg2_norm + environment_component + relatedness_component

        IID_FID = 'tsk_' + pd.DataFrame(range(phenotype_continuous.shape[1])).astype(str)

        phenotype_df = pd.concat([IID_FID, IID_FID, pd.DataFrame(phenotype_continuous.transpose())], axis=1)
        phenotype_df.columns = ['FID','IID', 'Y1']
        phenotype_df.to_csv('phenotype.csv', index=False, header=True, sep=' ')

        covariate_df = pd.concat([IID_FID, IID_FID,
                                  pd.DataFrame(environment_component.transpose()),
                                  pd.DataFrame(relatedness_component.transpose())
                                 ], axis=1)
        covariate_df.columns = ['FID','IID', 'V1', 'V2']
        covariate_df.to_csv('covariate.csv', index=False, header=True, sep=' ')

        environment_componentt=environment_component.transpose()
        relatedness_componentt=relatedness_component.transpose()

        con = np.hstack((con, environment_componentt))
        con = np.hstack((con, relatedness_componentt))
        print(con.shape)

        df = pd.DataFrame(con)

        print(df.tail())

        #name = "matrix_row_%d_col_%d.fp32_recombination8.bin" % (con.shape[0], con.shape[1])
        name = "genotype%d_%d.bin" % (int(sample), num_snps+2)
        f_output = open(name,"wb")

        for row in df.iterrows():
            index, data = row
            values = array('f',data)
            #print values
            values.tofile(f_output)

        f_output.close()

        phenotype_con = phenotype_continuous
        print(phenotype_con.shape)

        df = pd.DataFrame(phenotype_con)

        print(df.head())

        #name = "phenotype_row_%d_col_%d.fp32_recombination8.bin" % (phenotype_con.shape[0], phenotype_con.shape[1])
        name = "phenotype%d_%d.bin" % (int(sample), num_snps+2)
        f_output = open(name,"wb")

        for row in df.iterrows():
            index, data = row
            values = array('f',data)
            #print values
            values.tofile(f_output)

        f_output.close()

generate_and_save_matrix(int(filenumber))

