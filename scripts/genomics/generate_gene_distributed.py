import os
import msprime
from math import sqrt
import numpy as np
import pandas as pd
from array import array
import random
import argparse
import time
import queue
import concurrent.futures as cf
import pickle
import socket

from typing import List

from mpi4py import MPI


import torch
import torch.distributed as dist

#sample=60000
#will generate 30k patients and number of snpa determined by the msprime
#300k-

def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    
    # treat trivial case first
    if num_chunks == 1:
        return [size]
    
    # first, check if we can split using div-up to balance the load: 
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks-1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] >= num_chunks), f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
                                              {num_chunks} chunks. Empty slices are currently not supported."
    
    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)
    
    return tensor_list


def distributed_transpose(tensor, dim0, dim1, dim1_split_sizes, group=None):
    # get comm params
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    # split and local transposition
    tsplit = split_tensor_along_dim(tensor, num_chunks=comm_size, dim=dim0)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])

    # compute recv tensor
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))
    
    # global transposition
    dist.all_to_all(x_recv, x_send, group=group, async_op=False)

    # get dim0 split sizes
    x_recv = torch.cat(x_recv, dim=dim1)
    
    return x_recv


def save_files(output_directory, file_number, df_geno, df_pheno):
    geno_name = os.path.join(output_directory, f"genotype{file_number}.bin")
    pheno_name = os.path.join(output_directory, f"phenotype{file_number}.bin")

    # write pheno file
    with open(pheno_name,"wb") as f_output:
        df_pheno.to_numpy(dtype=np.float32).tofile(f_output)
        
    # write geno_file
    with open(geno_name,"wb") as f_output:
        df_geno.to_numpy(dtype=np.float32).tofile(f_output)

    return file_number


def select_snps(num_sites):
    if num_sites < 30000:
        num_snps = 204800-2
    elif (num_sites < 40000 and num_sites > 30000):
        num_snps = 30720-2
    elif (num_sites < 60000 and num_sites > 40000):
        num_snps = 51200-2
    elif (num_sites < 80000 and num_sites > 60000):
        num_snps = 61440-2
    elif (num_sites < 100000 and num_sites > 80000):
        num_snps = 81920-2
    elif (num_sites < 120000 and num_sites > 100000):
        num_snps = 102400-2
    elif (num_sites < 140000 and num_sites > 120000):
        num_snps = 122880-2 
    elif (num_sites < 160000 and num_sites > 140000):
        num_snps = 143360-2
    elif (num_sites < 180000 and num_sites > 160000):
        num_snps = 163840-2
    elif (num_sites < 200000 and num_sites > 180000):
        num_snps = 184320-2
    elif (num_sites < 220000 and num_sites > 200000):
        num_snps = 204800-2
    elif (num_sites < 240000 and num_sites > 220000):
        num_snps = 225280-2
    elif (num_sites < 260000 and num_sites > 240000):
        num_snps = 245760-2
    elif (num_sites < 280000 and num_sites > 260000):
        num_snps = 266240-2
    elif (num_sites < 300000 and num_sites > 280000):
        num_snps = 286720-2
    elif (num_sites < 500000 and num_sites > 300000): #1e7
        num_snps = 307200-2
    elif (num_sites < 800000 and num_sites > 500000): #2e7
        num_snps = 614400-2
    elif (num_sites < 1200000 and num_sites > 700000): #3e7
        num_snps = 1024000-2
    elif (num_sites < 3000000 and num_sites > 1200000): #4e7
        num_snps = 2048000-2
    elif (num_sites < 5000000 and num_sites > 3000000):#1e8
        num_snps = 3072000-2
    elif (num_sites < 8000000 and num_sites > 5000000): #
        num_snps = 6144000-2
    elif (num_sites < 12000000 and num_sites > 8000000):
        num_snps = 10240000-2
    elif (num_sites < 16000000 and num_sites > 12000000):
        num_snps = 12288000-2

    return num_snps

    

def generate_and_save_matrix(output_directory, filelist, num_samples, lens, comm, device, executor, max_queue_size, write_helper_files):

    ## determine lens:
    #lens = 1e7
    #if (num_samples >= 600000):
    #    lens = 2e7
    #if (num_samples >= 1000000):
    #    lens = 3e7

    # get communicator properties
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    print(f"Rank {comm_rank}: Generating files {filelist} with {num_samples} samples and length {lens}.", flush=True)

    # initialize rng
    seed = 333 + comm_rank

    # numpy rng
    r_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed=seed)

    # torch rng
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # requests queue
    requests = queue.Queue(maxsize=max_queue_size)
    for file_number in filelist:
        
        print(f"Rank {comm.Get_rank()}: Running msprime", flush=True)
        start_time = time.perf_counter_ns()
        ts = msprime.simulate(
            sample_size=2*int(num_samples),
            Ne=1e4,
            length=lens,
            recombination_rate=2e-8,
            mutation_rate=1e-7,
            random_seed = seed + 4553
        )

        num_snps = select_snps(ts.num_sites)        

        if write_helper_files:
            ofile = os.path.join(output_directory, f"simulation{file_number}.vcf")
            with open(ofile, "w") as vcf_file:
                ts.write_vcf(vcf_file, ploidy=2)

        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) * 10**(-9)
        print(f"Rank {comm_rank}: Msprime finished in {duration} s", flush=True)

        print(f"Rank {comm_rank}: Total number of SNPs {ts.num_sites}", flush=True)
        print(f"Rank {comm_rank}: Number of truncated SNPs: {num_snps}", flush=True)

        # Generate the genotype matrix with 0, 1, 2 values by summing the two alleles
        print(f"Rank {comm_rank}: Starting Genotype matrix", flush=True)
        start_time = time.perf_counter_ns()
        # truncated genotype matrix
        tsg2 = ts.genotype_matrix()

        # find minimum number of SNP across ranks
        num_snp_min = comm.allreduce(tsg2.shape[0], op=MPI.MIN)

        if num_snp_min < num_snps:
            raise RuntimeError("Error, the number of SNP is smaller than requested number of SNP. Please increase the length.")
        
        tsg2 = tsg2[:num_snps, :]

        # move to gpu
        tsg2_d = torch.from_numpy(tsg2).to(dtype=torch.float32, device=device)
        
        # sum of two components
        tsg2_d = tsg2_d.reshape(tsg2_d.shape[0], tsg2_d.shape[1]//2, 2)
        tsg2_d = torch.sum(tsg2_d, axis=-1)

        # globally transpose the data using all to all: currently, NPAT (dim=1) is split and NSNP(dim=0) is local, we want to make
        # dim 1 local (NPAT) and 0 split (NSNP)
        split_shapes = [tsg2_d.shape[1] for i in range(comm_size)]
        tsg2t_d = distributed_transpose(tsg2_d, dim0=0, dim1=1, dim1_split_sizes=split_shapes)

        # compute statistics:
        row_stds, row_means = torch.std_mean(tsg2t_d, dim=1, correction=0, keepdim=True)
        
        # regularize the row_stds:
        row_stds = torch.clamp(row_stds, min=1e-6)

        # save matrix so far on host
        con = tsg2_d.T.cpu().numpy()
        
        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) * 10**(-9)
        print(f"Rank {comm_rank}: Genotype matrix finished in {duration} s", flush=True)

        # some parameters
        print(f"Rank {comm_rank}: Starting normalization", flush=True)
        start_time = time.perf_counter_ns()
        genetic_variance = .7 # proportion of phenotypic variance explained by the genotypic variation
        environmental_variance = .15 # proportion of phenotypic variance explained by the environment
        relatedness_variance = .15 # proportion of phenotypic variance explained by population stratification

        # mean-variance normalization
        tsg2t_norm_d = (tsg2t_d - row_means) / row_stds
        
        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) * 10**(-9)
        print(f"Rank {comm_rank}: Normalization finished in {duration} s", flush=True)
        

        # We consider the setup where 10% of the SNPs are causal
        print(f"Rank {comm_rank}: Starting PCA", flush=True)
        start_time = time.perf_counter_ns()

        # generate causal SNP
        num_local_snp = tsg2t_norm_d.shape[0]
        num_causal_SNPs = num_local_snp // 10
        
        #pos_causal_SNPs = np.array(r_rng.sample(range(num_local_snp), num_causal_SNPs))
        # there is no random sample, so permute and choose first few:
        pos_causal_SNPs = torch.randperm(num_local_snp, device=device)[:num_causal_SNPs]
        pos_causal_SNPs, _ = torch.sort(pos_causal_SNPs)
        
        weight_causal_SNPs = torch.zeros((1, tsg2t_norm_d.shape[0]), dtype=torch.float32, device=device)
        weight_causal_SNPs[:,pos_causal_SNPs] = sqrt(genetic_variance / num_causal_SNPs) * torch.randn((num_causal_SNPs), dtype=torch.float32, device=device)

        environment_component = sqrt(environmental_variance) * torch.randn((1, tsg2t_norm_d.shape[1]), dtype=torch.float32, device=device)

        
        # do distributed PCA according to https://www.cs.cmu.edu/~ninamf/papers/distributedPCAandCoresets.pdf
        # compute sample covariance matrix, just the first component
        num_comp = 1
        _, Sproj, Vproj = torch.pca_lowrank(tsg2t_norm_d, q=num_comp, center=False, niter=4)

        # gather components:
        # S
        #Sproj_gather = [torch.empty_like(Sproj) for _ in range(comm_size)]
        #dist.all_gather(Sproj_gather, Sproj)
        # V
        #Vproj_gather = [torch.empty_like(Vproj)	for _ in range(comm_size)]
        #dist.all_gather(Vproj_gather, Vproj)
        # Compute S
        #SVproj_gather = [torch.matmul(torch.diag(S), V.T) for S, V in zip(Sproj_gather, Vproj_gather)]
        #Smat = sum([torch.matmul(P.T, P) for P in SVproj_gather])
        # Diagonalize
        #_, V = torch.linalg.eig(Smat)
        #Vproj = V[:,:num_comp]
        
        # we can save some GPU memory here
        P = torch.matmul(torch.diag(Sproj), Vproj.T).contiguous()
        P_gather = [torch.empty_like(P) for _ in range(comm_size)]
        
        # gather
        dist.all_gather(P_gather, P)
        Pfull = torch.cat(P_gather, dim=0)
        
        # PCA
        _, _, Vproj = torch.pca_lowrank(Pfull, q=num_comp, center=True, niter=4)

        # we have to transpose Vproj, because how the eigensystem is defined
        relatedness_component = Vproj.T
        
        relatedness_component = relatedness_component / torch.std(relatedness_component, dim=1) * sqrt(relatedness_variance)
        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) * 10**(-9)
        print(f"Rank {comm_rank}: PCA finshed in {duration} s", flush=True)
        

        # Continous outcome
        print(f"Rank {comm_rank}: Starting phenotype calculation", flush=True)
        start_time = time.perf_counter_ns()

        # do distributes matmul:
        # local path
        phenotype_continuous = torch.matmul(weight_causal_SNPs, tsg2t_norm_d)
        # reduction across nodes
        dist.all_reduce(phenotype_continuous)
        
        # transpose the data back, every rank only takes his chunk in npat:
        pat_split_shapes = compute_split_shapes(environment_component.shape[1], comm_size)
        environment_component = torch.split(environment_component, pat_split_shapes, dim=1)[comm_rank]
        relatedness_component = torch.split(relatedness_component, pat_split_shapes, dim=1)[comm_rank]
        phenotype_continuous = torch.split(phenotype_continuous, pat_split_shapes, dim=1)[comm_rank]
        snp_split_shapes = comm.allgather(tsg2t_norm_d.shape[0])
        tsg2_norm_d = distributed_transpose(tsg2t_norm_d, dim0=1, dim1=0, dim1_split_sizes=snp_split_shapes)

        # add missing pieces
        phenotype_continuous = phenotype_continuous + environment_component + relatedness_component
        
        IID_FID = 'tsk_' + pd.DataFrame(range(phenotype_continuous.shape[1])).astype(str)

        phenotype_df = pd.concat([IID_FID, IID_FID, pd.DataFrame(phenotype_continuous.cpu().numpy().transpose())], axis=1)
        phenotype_df.columns = ['FID','IID', 'Y1']

        # we do not need to write that
        #ofile = os.path.join(output_directory, f"phenotype{file_number}.csv")
        #phenotype_df.to_csv(ofile, index=False, header=True, sep=' ')

        environment_componentt=environment_component.cpu().numpy().transpose()
        relatedness_componentt=relatedness_component.cpu().numpy().transpose()
        covariate_df = pd.concat([IID_FID, IID_FID, 
                                  pd.DataFrame(environment_componentt),
                                  pd.DataFrame(relatedness_componentt)
                                 ], axis=1)
        covariate_df.columns = ['FID','IID', 'V1', 'V2']
        
        #ofile = os.path.join(output_directory, f"covariate{file_number}.csv")
        #covariate_df.to_csv(ofile, index=False, header=True, sep=' ')

        con = np.hstack((con, environment_componentt))
        con = np.hstack((con, relatedness_componentt))

        df_geno = pd.DataFrame(con)
        df_pheno = pd.DataFrame(phenotype_continuous.cpu().numpy())
        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) * 10**(-9)
        print(f"Rank {comm_rank}: Phenotype calculation finshed in {duration} s", flush=True)


        print(f"Rank {comm_rank}: Submitting writing files number {file_number}", flush=True)
        start_time = time.perf_counter_ns()
        while requests.full():
            print(f"Rank {comm_rank}: Waiting for queue to catch up", flush=True)
            req = requests.get()
            fn = req.result()
            print(f"Rank {comm_rank}: File number {fn} done, continue", flush=True)
            break
        requests.put(executor.submit(save_files, output_directory, file_number, df_geno.copy(), df_pheno.copy()))
        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) * 10**(-9)
        print(f"Rank {comm_rank}: Submitting writing files finshed in {duration} s", flush=True)

        # better wait before starting next file
        comm.Barrier()

    # wait for queue to finish:
    while not requests.empty():
        req = requests.get()
        req.result()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str, help="Directory where to put the output files", required=True)
    parser.add_argument("--number_of_files", default=1, type=int, help="Number of files to generate")
    parser.add_argument("--number_of_links", default=0, type=int, help="Number of virtual files (links) to generate, drawn from the other set")
    parser.add_argument("--number_of_samples", default=2048, type=int, help="Number of samples/patients per file (multiple of tile size)")
    parser.add_argument("--length_of_sequence", default=1e7, type=float, help="Length")
    parser.add_argument("--num_writers", default=1, type=int, help="Number of writing processes")
    parser.add_argument("--max_io_queue_size", default=2, type=int, help="Number of queued IO operations")
    parser.add_argument("--write_helper_files", action='store_true')
    args = parser.parse_args()

    # initialize MPI
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # set env variables for torch distributed
    os.environ["MASTER_PORT"] = "29500"
    
    # initialize torch distributed
    dist.init_process_group(backend='nccl',
                            world_size=comm_size,
                            rank=comm_rank)
    
    # set device
    comm_local_rank = comm_rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{comm_local_rank}")
    torch.cuda.set_device(device)

    # bulk
    # make sure the number of files is an integer multiple of the number of ranks
    if args.number_of_files % comm_size != 0:
        raise ValueError("Number of files must be divisible by the number of ranks")
    number_of_files_per_rank = args.number_of_files // comm_size
    files_start = number_of_files_per_rank * comm_rank
    files_end = files_start + number_of_files_per_rank
    filelist = list(range(files_start, files_end))

    # create output directory
    if comm_rank == 0:
        os.makedirs(args.output_directory, exist_ok=True)
    comm.Barrier()

    # create executor:
    executor = cf.ProcessPoolExecutor(max_workers=args.num_writers)
    
    # generate data
    with torch.no_grad():
        generate_and_save_matrix(args.output_directory, filelist, args.number_of_samples, args.length_of_sequence,
                                 comm, device, executor, args.max_io_queue_size, args.write_helper_files)

    # shutdown executor
    executor.shutdown()

    # wait for everybody to finish
    comm.Barrier()

    # now generate links
    if args.number_of_links > 0:
        r_rng = random.Random(333)
        resample = r_rng.choices(list(range(args.number_of_files)), k=args.number_of_links)

        # chunk across ranks round robin
        my_resample = []
        for idr, rs in enumerate(resample):
            if (idr % comm_size == comm_rank):
                my_resample.append((idr, rs))

        # create links
        for idr, rs in my_resample:
            # get total file index
            idt = idr + args.number_of_files
            # genotypes:
            source_name = os.path.join(args.output_directory, f"genotype{rs}.bin")
            dest_name = os.path.join(args.output_directory, f"genotype{idt}.bin")
            os.symlink(source_name, dest_name)
            # phenotypes
            source_name = os.path.join(args.output_directory, f"phenotype{rs}.bin")
            dest_name = os.path.join(args.output_directory, f"phenotype{idt}.bin")
            os.symlink(source_name, dest_name)

        # wait for others
        comm.Barrier()

    # wait for everybody
    dist.barrier(device_ids=[device.index])

    # clean up
    dist.destroy_process_group()
