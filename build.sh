#!/bin/bash

# Build Docker image
echo "Building Docker image..."
docker build -t cuda-dev:latest .

# Compile CUDA programs
echo "Building CUDA programs..."
docker run --rm -v "$(pwd)":/workspace cuda-dev:latest bash -c "
  echo 'Compiling hello.cu...'
  nvcc -o hello hello.cu
  
  echo 'Compiling vector_add.cu...'
  nvcc -o vector_add vector_add.cu

  echo 'Compiling sales_report.cu...'
  nvcc -o sales_report sales_report.cu

  echo 'Build complete!'
  ls -la hello vector_add sales_report
"

# Run sales_report
echo ""
echo "Running sales_report..."
docker run --rm -v "$(pwd)":/workspace cuda-dev:latest ./sales_report

echo "Done!"
