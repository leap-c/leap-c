#!/bin/bash

# Exit on any error
set -e

# Script to update and build acados with OpenMP support
echo "Starting acados build process..."

# Update git submodules
git submodule update
if [ $? -ne 0 ]; then
    echo "Error: Failed to update git submodules"
    exit 1
fi

# Navigate to acados directory
cd external/acados || exit 1

# Remove existing build directory and create new one
rm -rf build
mkdir -p build
cd build || exit 1

# Run cmake configuration
echo "Configuring cmake..."
cmake .. -DACADOS_WITH_OPENMP=ON -DACADOS_PYTHON=ON -DACADOS_NUM_THREADS=1 || {
    echo "Error: CMAKE configuration failed"
    exit 1
}

# Build and install
echo "Building and installing..."
make install -j6 || {
    echo "Error: Make install failed"
    exit 1
}

# Return to original directory
cd ../../..

echo "Build process completed successfully!"
