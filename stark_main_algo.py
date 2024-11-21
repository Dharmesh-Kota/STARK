import numpy as np
import math
import time
from concurrent.futures import ThreadPoolExecutor

def load_matrix_1(size, path):
    matrix_df = spark.read.parquet(f"{path}/matrix_{size}x{size}_1.parquet")
    
    matrix = np.array(matrix_df.select("row").rdd.map(lambda row: row[0]).collect())
    return matrix

def load_matrix_2(size, path):
    matrix_df = spark.read.parquet(f"{path}/matrix_{size}x{size}_2.parquet")
    
    matrix = np.array(matrix_df.select("row").rdd.map(lambda row: row[0]).collect())
    return matrix

# Matrix loading and paramters setting
matrix_size = 2048
num_splits = 2
block_size = matrix_size // num_splits
base_path = "/mnt/matrices"
A = load_matrix_1(matrix_size, base_path)
B = load_matrix_2(matrix_size, base_path)

# Divide and Replication Phase
def divide_and_replicate(rdd_matrix, matrix_name='A'):
    def divide_block(block_data):
        ((row_start, col_start), matrix) = block_data
        n = matrix.shape[0] // 2

        top_left = matrix[:n, :n]
        top_right = matrix[:n, n:]
        bottom_left = matrix[n:, :n]
        bottom_right = matrix[n:, n:]

        return [
            (f'{matrix_name}11', (row_start, col_start), top_left),
            (f'{matrix_name}12', (row_start, col_start + n), top_right),
            (f'{matrix_name}21', (row_start + n, col_start), bottom_left),
            (f'{matrix_name}22', (row_start + n, col_start + n), bottom_right)
        ]

    divided_blocks_rdd = rdd_matrix.flatMap(divide_block)

    replication_map = {
        'A': {
            'A11': [1, 3, 5, 6],
            'A12': [5, 7],
            'A21': [2, 6],
            'A22': [1, 2, 4, 7]
        },
        'B': {
            'B11': [1, 2, 4, 6],
            'B12': [3, 6],
            'B21': [4, 7],
            'B22': [1, 3, 5, 7]
        }
    }

    def replicate_with_metadata(submatrix):
        submatrix_name, (row_index, col_index), block = submatrix
        m_indices = replication_map[matrix_name].get(submatrix_name, [])

        replicated_data = [
            (m_index, ({
                'm_index': m_index,
                'row_index': row_index,
                'col_index': col_index,
                'matrix_name': f"{submatrix_name}, M{m_index}"
            }, block)) 
            for m_index in m_indices
        ]
        return replicated_data

    replicated_with_metadata = divided_blocks_rdd.flatMap(replicate_with_metadata)
    
    return replicated_with_metadata

# Recursive Strassen function
def strassen_recursive(rdd_A, rdd_B, block_size, current_size, parentIndex=0):
    if current_size <= block_size:
        mat_A = rdd_A.collect()[0][1]
        mat_B = rdd_B.collect()[0][1]
        result = mat_A @ mat_B
        return spark.sparkContext.parallelize([((parentIndex, parentIndex), result)])

    rdd_combined = divide_and_replicate(rdd_A, matrix_name='A') \
        .union(divide_and_replicate(rdd_B, matrix_name='B'))
    # Use this snippet if you have a distributed environment instead of using the above
    # rdd_A_rep = divide_and_replicate(rdd_A, block_size, matrix_name='A').persist()
    # rdd_B_rep = divide_and_replicate(rdd_B, block_size, matrix_name='B').persist()
    # rdd_combined = rdd_A_rep.union(rdd_B_rep)

    rdd_grouped = rdd_combined.groupByKey().mapValues(list)

    def get_matrices(m_index, blocks):
        """Creates mat1 and mat2 based on m_index for Strassen's operations."""
        sub_blocks = {entry[0]['matrix_name']: entry[1] for entry in blocks}
        if m_index == 1:
            mat1 = add_blocks(sub_blocks['A11, M1'], sub_blocks['A22, M1'])
            mat2 = add_blocks(sub_blocks['B11, M1'], sub_blocks['B22, M1'])
        elif m_index == 2:
            mat1 = add_blocks(sub_blocks['A21, M2'], sub_blocks['A22, M2'])
            mat2 = sub_blocks['B11, M2']
        elif m_index == 3:
            mat1 = sub_blocks['A11, M3']
            mat2 = subtract_blocks(sub_blocks['B12, M3'], sub_blocks['B22, M3'])
        elif m_index == 4:
            mat1 = sub_blocks['A22, M4']
            mat2 = subtract_blocks(sub_blocks['B21, M4'], sub_blocks['B11, M4'])
        elif m_index == 5:
            mat1 = add_blocks(sub_blocks['A11, M5'], sub_blocks['A12, M5'])
            mat2 = sub_blocks['B22, M5']
        elif m_index == 6:
            mat1 = subtract_blocks(sub_blocks['A21, M6'], sub_blocks['A11, M6'])
            mat2 = add_blocks(sub_blocks['B11, M6'], sub_blocks['B12, M6'])
        elif m_index == 7:
            mat1 = subtract_blocks(sub_blocks['A12, M7'], sub_blocks['A22, M7'])
            mat2 = add_blocks(sub_blocks['B21, M7'], sub_blocks['B22, M7'])
        return mat1, mat2

    M_rdds = []
    for m_index, blocks in rdd_grouped.collect():
        mat1, mat2 = get_matrices(m_index, blocks)
        new_parentIndex = parentIndex * 7 + m_index

        rdd_mat1 = spark.sparkContext.parallelize([((m_index, new_parentIndex), mat1)])
        rdd_mat2 = spark.sparkContext.parallelize([((m_index, new_parentIndex), mat2)])

        M_rdds.append(strassen_recursive(rdd_mat1, rdd_mat2, block_size, current_size // 2, new_parentIndex))

    M_results = {m_index + 1: rdd.collect()[0][1] for m_index, rdd in enumerate(M_rdds)}

    C11 = add_blocks(add_blocks(M_results[1], M_results[4]), subtract_blocks(M_results[7], M_results[5]))
    C12 = add_blocks(M_results[3], M_results[5])
    C21 = add_blocks(M_results[2], M_results[4])
    C22 = add_blocks(subtract_blocks(M_results[1], M_results[2]), add_blocks(M_results[3], M_results[6]))

    # Use this snippet if you have a distributed environment instead of using the above
    # with ThreadPoolExecutor() as executor:
    #     future_C11 = executor.submit(lambda: add_blocks(add_blocks(M_results[1], M_results[4]), subtract_blocks(M_results[7], M_results[5])))
    #     future_C12 = executor.submit(lambda: add_blocks(M_results[3], M_results[5]))
    #     future_C21 = executor.submit(lambda: add_blocks(M_results[2], M_results[4]))
    #     future_C22 = executor.submit(lambda: add_blocks(subtract_blocks(M_results[1], M_results[2]), add_blocks(M_results[3], M_results[6])))

    # C11 = future_C11.result()
    # C12 = future_C12.result()
    # C21 = future_C21.result()
    # C22 = future_C22.result()

    # Combine C11, C12, C21, and C22 into a single matrix
    combined_matrix = np.block([
        [C11, C12],
        [C21, C22]
    ])

    return spark.sparkContext.parallelize([((parentIndex, parentIndex), combined_matrix)])

def add_blocks(block1, block2):
    return block1 + block2

def subtract_blocks(block1, block2):
    return block1 - block2

rdd_A = sc.parallelize([((0, 0), A)])
rdd_B = sc.parallelize([((0, 0), B)])

start_time = time.time()
result_rdd = strassen_recursive(rdd_A, rdd_B, block_size, current_size=A.shape[0])
end_time = time.time()
duration = end_time - start_time

print(result_rdd.collect())
print(duration)