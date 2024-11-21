# STARK: Fast and Scalable Strassen’s Matrix Multiplication Using Apache Spark

## Overview
This project implements a parallel version of Strassen's matrix multiplication algorithm using Apache Spark, focusing on scalability and performance in distributed environments. The code is designed for efficient matrix multiplication in Spark using recursive decomposition, optimized data handling, and parallel processing to minimize communication overhead. The project simulates matrix multiplication on Databricks and provides performance comparisons with Spark MLlib and NumPy.

## Features
- *Matrix Generation and Storage*: Generates random matrices of varying sizes, converts them to Spark DataFrames, and stores them in Parquet format for efficient I/O operations.
- *Recursive Strassen’s Multiplication*: A recursive implementation of Strassen’s matrix multiplication optimized for parallel execution on distributed clusters.
- *Matrix Loading and Partitioning*: Loads matrices from Parquet files and partitions them for Strassen’s recursive subproblems.
- *Comparison with MLlib and NumPy*: Performance evaluation using both distributed (MLlib) and local (NumPy) matrix multiplication for benchmark analysis.

## Setup
### Requirements
- Databricks Runtime Version 12.2 LTS (Apache Spark 3.3.2, Scala 2.12)
- Python libraries: numpy, pyspark
  
### Cluster Configuration
The code is optimized for Databricks Community Edition. For larger datasets, increase the cluster resources (e.g., memory, CPU cores) to improve performance.

### Directory Setup
Ensure that the directory /mnt/matrices in DBFS is accessible for storing generated matrices.

## Project Structure
- *Matrix Generation*: Clears any existing matrices in /mnt/matrices and generates random matrices of various sizes, saving them in Parquet format.
- *Matrix Loading and Multiplication*:
  - *Loading*: Loads matrices from Parquet and prepares them for Strassen’s algorithm.
  - *Strassen’s Algorithm*: A recursive function that divides, replicates, and combines matrices, leveraging Spark’s parallel processing.
- *Performance Comparison*: Measures execution time for distributed multiplication with MLlib and local multiplication using NumPy, highlighting performance efficiency.

## Usage
1. *Matrix Generation*: Run the generate_and_save_matrix functions to create matrices of sizes from \(2^3\) to \(2^{12}\).
2. *Matrix Multiplication*: Use strassen_recursive to perform Strassen’s multiplication on large matrices.
3. *Benchmark*: Run MLlib and NumPy comparison functions to analyze and compare execution times.

## Example Usage
The example notebook demonstrates matrix generation, loading, recursive multiplication, and timing comparisons. To run:
- Clone this repository into your Databricks workspace.
- Execute the cells in order to generate matrices, load them, and compute their product using Strassen’s algorithm.