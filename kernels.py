"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE. 
    Be careful which one indexes the column.
    """

    # need to get rows and cols global
    # we know that the grid is only 1D and blockIdx.x runs 0 -> 1023
    # blockIdx.x refers to the 
    row_in_tile = cuda.threadIdx.x // BLOCKSIZE
    col_in_tile = cuda.threadIdx.x % BLOCKSIZE

    # the method to find global rows and columns stay the same
    # basically (which block) * (how big the block is) * (your position in block)
    row = cuda.blockIdx.x * BLOCKSIZE + row_in_tile
    col = cuda.blockIdx.y * BLOCKSIZE + col_in_tile

    if row < M and col < N:
        acc = float32(0.0)
        for k in range(K):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc

    


# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """
    
    # need to get row and col in tiles
    row_in_tile = cuda.threadIdx.x // BN3
    col_in_tile = cuda.threadIdx.x % BN3


    # we need to create the shared memory arrays (A_shared, B_shared) and 
    # accumulator variable (float32(0.0))

    A_shared = cuda.shared.array((BM3, BK3), float32)
    B_shared = cuda.shared.array((BK3, BN3), float32)

    acc = float32(0.0)

    # output global values for C
    global_row = cuda.blockIdx.x * BM3 + row_in_tile
    global_col = cuda.blockIdx.y * BN3 + col_in_tile 

    # outer loop -> iterates through N number of chunks (K // BK3)
    for chunk in range((K+BK3-1) // BK3): 

        # need to determine globals for row and col
        global_row_A = cuda.blockIdx.x * BM3 + row_in_tile
        global_col_A = chunk * BK3 + col_in_tile # kth dimension 

        global_row_B = chunk * BK3 + row_in_tile # kth dimension
        global_col_B = cuda.blockIdx.y * BN3 + col_in_tile 


        # now we need to load into the shared memory from global memory
        if global_row_A < M and global_col_A < K:
            A_shared[row_in_tile, col_in_tile] = A[global_row_A, global_col_A]
        else:
            A_shared[row_in_tile, col_in_tile] = float32(0.0)
        if global_row_B < K and global_col_B < N:
            B_shared[row_in_tile, col_in_tile] = B[global_row_B, global_col_B]
        else:
            B_shared[row_in_tile, col_in_tile] = float32(0.0)

        # after loading into shared memory, sync before any reads/writes
        cuda.syncthreads()

        # inner loop for actual matmul
        for k in range(BK3):
            acc += A_shared[row_in_tile, k] * B_shared[k, col_in_tile] # row of A x col of b
        cuda.syncthreads()
    if global_row < M and global_col < N:
        C[global_row, global_col] = acc
    else:
        # if out of bounds, we don't write to C, but we still need to sync
        pass
    



# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
    
@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    thread_row = cuda.threadIdx.x // BN4
    thread_col = cuda.threadIdx.x % BN4
    # with the axis swap, blockIdx.x indexes columns of C
    # , so we compute the global row and col accordingly
    block_row = cuda.blockIdx.y * BM4
    block_col = cuda.blockIdx.x * BN4
    global_row = block_row + thread_row * TM4
    global_col = block_col + thread_col
    row_base = thread_row * TM4


    acc = cuda.local.array(TM4, float32)
    for i in range(TM4):
        acc[i] = float32(0.0) # initialize accumulators to 0.0

    A_shared = cuda.shared.array((BK4, BM4), float32)
    B_shared = cuda.shared.array((BK4, BN4), float32)
    # with the cooperative loads, each thread loads one element of A and one element of B into shared memory
    a_row = cuda.threadIdx.x // BK4
    a_col = cuda.threadIdx.x % BK4
    b_row = cuda.threadIdx.x // BN4
    b_col = cuda.threadIdx.x % BN4

    a_row_global = block_row + a_row
    b_col_global = block_col + b_col
    # we need to loop over K in chunks of BK4, and for each chunk, we cooperatively load the tile of A and B into shared memory, then do the updates for our register tile
    for chunk in range((K + BK4 - 1) // BK4):
        chunk_offset = chunk * BK4
        a_col_global = chunk_offset + a_col
        b_row_global = chunk_offset + b_row
        # for the cooperative load, we need to check bounds and load 0.0 if out of bounds
        if a_row_global < M and a_col_global < K:
            A_shared[a_col, a_row] = A[a_row_global, a_col_global]
        else:
            A_shared[a_col, a_row] = float32(0.0)
        if b_row_global < K and b_col_global < N:
            B_shared[b_row, b_col] = B[b_row_global, b_col_global]
        else:
            B_shared[b_row, b_col] = float32(0.0)

        cuda.syncthreads()
        # now we have the tile of A and B in shared memory, we can compute the updates for our register tile
        for k in range(BK4):
            b_tmp = B_shared[k, thread_col]
            for j in range(TM4):
                acc[j] += A_shared[k, row_base + j] * b_tmp
        cuda.syncthreads()
        # after the K-loop, we write our register tile back to global memory
    for j in range(TM4):
        if global_row + j < M and global_col < N:
            C[global_row + j, global_col] = acc[j]


# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    # we need to get the row and col of the thread in the tile
    thread_row = cuda.threadIdx.x // (BN5 // TN5)
    thread_col = cuda.threadIdx.x % (BN5 // TN5)

    # then we can get the global row and col of the tile start
    global_row = cuda.blockIdx.y * BM5 + thread_row * TM5
    global_col = cuda.blockIdx.x * BN5 + thread_col * TN5

    acc = cuda.local.array((TM5, TN5), float32) # accumulator array for each thread
    for i in range(TM5):
        for j in range(TN5):
            acc[i, j] = float32(0.0) # initialize accumulators to 0.0
    # shared memory arrays for A and B
    A_shared = cuda.shared.array((BM5, BK5), float32)
    B_shared = cuda.shared.array((BK5, BN5), float32)

    # compute the number of threads and the stride for loading A 
    # and B into shared memory
    num_threads = (BM5 * BN5) // (TM5 * TN5)
    a_stride = num_threads // BK5 # number of threads that will load A's tile, divided by the K dimension of the tile
    b_stride = num_threads // BN5 # number of threads that will load B's tile, divided by the N dimension of the tile

    for chunk in range((K + BK5 - 1) // BK5):
        for i in range(BM5 // a_stride):
            a_row = cuda.threadIdx.x // BK5 + i * a_stride
            a_col = cuda.threadIdx.x % BK5
            if cuda.blockIdx.y * BM5 + a_row < M and chunk * BK5 + a_col < K:
                A_shared[a_row, a_col] = A[cuda.blockIdx.y * BM5 + a_row, chunk * BK5 + a_col]
            else:
                A_shared[a_row, a_col] = float32(0.0)
        for i in range(BK5 // b_stride):
            b_row = cuda.threadIdx.x // BN5 + i * b_stride
            b_col = cuda.threadIdx.x % BN5
            if chunk * BK5 + b_row < K and cuda.blockIdx.x * BN5 + b_col < N:
                B_shared[b_row, b_col] = B[chunk * BK5 + b_row, cuda.blockIdx.x * BN5 + b_col]
            else:
                B_shared[b_row, b_col] = float32(0.0)

        cuda.syncthreads()
        # now we have the tile of A and B in shared memory,
        #  we can compute the outer product updates for our register tile
        for k in range(BK5):
            a_reg = cuda.local.array(TM5, float32)
            b_reg = cuda.local.array(TN5, float32)

            # load the values of A and B for this k into registers
            for i in range(TM5):
                a_reg[i] = A_shared[thread_row * TM5 + i, k]
            for j in range(TN5):
                b_reg[j] = B_shared[k, thread_col * TN5 + j]
            for i in range(TM5):
                for j in range(TN5):
                    acc[i, j] += a_reg[i] * b_reg[j]

        cuda.syncthreads()

    for i in range(TM5):
        for j in range(TN5):
            if global_row + i < M and global_col + j < N:
                C[global_row + i, global_col + j] = acc[i, j]


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]
