#!/bin/bash

# Script to install mbi and MP-SPDZ
# Author: Installation Script
# Date: 2025-11-15

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Starting Installation Process${NC}"
echo -e "${GREEN}======================================${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}>>> $1${NC}\n"
}

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 successful${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Store the starting directory (should be home)
START_DIR=$(pwd)
echo "Starting directory: $START_DIR"

#############################################
# INSTALL MBI
#############################################

print_section "Installing mbi"

# Install mbi via pip first
echo "Installing mbi package via pip..."
pip install mbi==1.0.0
check_success "mbi pip installation"

# Clone mbi repository
echo "Cloning mbi repository..."
if [ -d "mbi" ]; then
    echo "mbi directory already exists. Removing it..."
    rm -rf mbi
fi
git clone https://github.com/ryan112358/mbi.git
check_success "mbi repository clone"

# Install mbi from source
echo "Installing mbi from source..."
cd mbi
pip install .
check_success "mbi source installation"

# Install matplotlib
echo "Installing matplotlib..."
pip install matplotlib
check_success "matplotlib installation"

# Verify mbi folder exists
cd ..
if [ -d "mbi" ]; then
    echo -e "${GREEN}✓ mbi folder verified${NC}"
else
    echo -e "${RED}✗ mbi folder not found${NC}"
    exit 1
fi

# Check if we are in home folder (outside of mbi)
CURRENT_DIR=$(pwd)
if [ "$CURRENT_DIR" = "$START_DIR" ]; then
    echo -e "${GREEN}✓ Back in home folder: $CURRENT_DIR${NC}"
else
    echo -e "${YELLOW}Warning: Current directory is $CURRENT_DIR, expected $START_DIR${NC}"
    cd "$START_DIR"
    echo "Changed back to: $(pwd)"
fi

#############################################
# INSTALL MP-SPDZ
#############################################

print_section "Installing MP-SPDZ"

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt-get update
check_success "apt-get update"

echo "Upgrading system packages..."
sudo apt-get upgrade -y
check_success "apt-get upgrade"

# Install dependencies
echo "Installing MP-SPDZ dependencies..."
sudo apt-get install -y automake build-essential clang cmake git libboost-dev \
    libboost-thread-dev libgmp-dev libntl-dev libsodium-dev libssl-dev libtool
check_success "dependencies installation"

# Clone MP-SPDZ repository
echo "Cloning MP-SPDZ repository..."
if [ -d "MP-SPDZ" ]; then
    echo "MP-SPDZ directory already exists. Removing it..."
    rm -rf MP-SPDZ
fi
git clone https://github.com/data61/MP-SPDZ.git
check_success "MP-SPDZ repository clone"

# Build MP-SPDZ
cd MP-SPDZ/
echo "Building MP-SPDZ components..."

echo "Making cmake..."
make cmake
check_success "cmake build"

echo "Making boost..."
make boost
check_success "boost build"

echo "Making replicated-ring-party.x..."
make replicated-ring-party.x
check_success "replicated-ring-party.x build"

# Setup SSL
echo "Setting up SSL for 4 parties..."
Scripts/setup-ssl.sh 4
check_success "SSL setup"

# Return to home directory
cd "$START_DIR"

#############################################
# VERIFICATION TESTS
#############################################

print_section "Running Verification Tests"

# Test mbi
echo "Testing mbi installation..."
cd mbi/mechanisms
python3 aim.py
check_success "mbi test (aim.py)"
cd "$START_DIR"

# Test MP-SPDZ
echo "Testing MP-SPDZ installation..."
cd MP-SPDZ

echo "Compiling tutorial..."
./compile.py -R 64 tutorial
check_success "tutorial compilation"

echo "Creating test input files..."
mkdir -p Player-Data
echo "1 2 3 4" > Player-Data/Input-P0-0
echo "1 2 3 4" > Player-Data/Input-P1-0
check_success "input file creation"

echo "Running tutorial with ring.sh..."
Scripts/ring.sh tutorial
check_success "tutorial execution"

cd "$START_DIR"

#############################################
# COMPLETION
#############################################

print_section "Installation Complete!"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "${GREEN}  ✓ mbi installed and verified${NC}"
echo -e "${GREEN}  ✓ MP-SPDZ installed and verified${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "\nInstallation directories:"
echo -e "  mbi: $START_DIR/mbi"
echo -e "  MP-SPDZ: $START_DIR/MP-SPDZ"
echo -e "\nYou are currently in: $(pwd)"


#############################################
# Updating Files for CaPS
#############################################

print_section "Updating files for CaPS -- TBD"
# Will upload files to GitHub, and you can copy them from there
