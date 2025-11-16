#!/usr/bin/env python3
"""
Complete Environment Verification Script
"""

import importlib
import sys
import subprocess
from pathlib import Path

def check_package(package_name, import_name=None, min_version=None):
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        
        if min_version and version != 'Unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"{package_name}: {version} (requires >= {min_version})")
                return False
        
        print(f"{package_name}: {version}")
        return True
        
    except ImportError as e:
        print(f"{package_name}: Not installed - {e}")
        return False

def check_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"CUDA: Available (Version: {cuda_version})")
            print(f"GPU: {device_name} (Count: {device_count})")
            
            # 测试GPU张量
            x = torch.randn(3, 3).cuda()
            print(f"GPU Tensor Test: {x.device}")
            return True
        else:
            print("CUDA: Not available")
            return False
            
    except Exception as e:
        print(f"CUDA Check Failed: {e}")
        return False

def check_pyg():
    pyg_packages = [
        ("torch-scatter", "torch_scatter"),
        ("torch-sparse", "torch_sparse"), 
        ("torch-cluster", "torch_cluster"),
        ("torch-spline-conv", "torch_spline_conv"),
        ("PyTorch Geometric", "torch_geometric")
    ]
    
    all_ok = True
    for name, import_name in pyg_packages:
        if not check_package(name, import_name):
            all_ok = False
    
    return all_ok

def check_nlp_packages():
    nlp_packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("transformers", "transformers"),
        ("NLTK", "nltk"),
        ("spaCy", "spacy")
    ]
    
    all_ok = True
    for name, import_name in nlp_packages:
        if not check_package(name, import_name):
            all_ok = False
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("spaCy Model: en_core_web_sm loaded successfully")
    except Exception as e:
        print(f"spaCy Model: Failed to load en_core_web_sm - {e}")
        all_ok = False
    
    return all_ok

def check_data_science_packages():
    ds_packages = [
        ("pandas", "pandas", "1.5.0"),
        ("numpy", "numpy", "1.23.0"),
        ("scipy", "scipy", "1.9.0"),
        ("scikit-learn", "sklearn", "1.2.0"),
        ("matplotlib", "matplotlib", "3.6.0"),
        ("seaborn", "seaborn", "0.12.0"),
        ("plotly", "plotly", "5.13.0")
    ]
    
    all_ok = True
    for name, import_name, min_version in ds_packages:
        if not check_package(name, import_name, min_version):
            all_ok = False
    
    jupyter_packages = [
        ("notebook", "notebook", "6.0.0"),
        ("jupyterlab", "jupyterlab", "3.0.0"),
        ("ipykernel", "ipykernel", "6.0.0"),
        ("jupyter_client", "jupyter_client", "7.0.0")
    ]
    
    print("\n Checking Jupyter Packages...")
    for name, import_name, min_version in jupyter_packages:
        if not check_package(name, import_name, min_version):
            all_ok = False
    
    return all_ok

def check_graph_packages():
    graph_packages = [
        ("networkx", "networkx", "3.0.0"),
        ("DGL", "dgl", "1.1.0"),
        ("py2neo", "py2neo", "2021.2.3")
    ]
    
    all_ok = True
    for name, import_name, min_version in graph_packages:
        if not check_package(name, import_name, min_version):
            all_ok = False
    
    return all_ok

def check_package(package_name, import_name=None, min_version=None):
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        
        # 检查最低版本要求
        if min_version and version != 'Unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠️  {package_name}: {version} (requires >= {min_version})")
                return False
        
        print(f"✅ {package_name}: {version}")
        return True
        
    except ImportError as e:
        print(f"{package_name}: Not installed - {e}")
        return False

def check_patool_special():
    try:
        import patoolib
        version = getattr(patoolib, '__version__', 'Unknown')
        print(f"✅ patool: {version}")
        return True
    except ImportError as e:
        print(f"patool: Import failed - {e}")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "show", "patool"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"pip shows patool is installed, but import fails")
                print(f"This might be a path issue. Location info:")
                for line in result.stdout.split('\n'):
                    if 'Location' in line:
                        print(f"   {line}")
        except:
            pass
        return False

def check_utility_packages():
    print("\n🔧 Checking Utility Packages...")
    
    utility_packages = [
        ("requests", "requests", "2.28.0"),
        ("beautifulsoup4", "bs4"),
        ("lxml", "lxml", "4.9.0"),
        ("tqdm", "tqdm", "4.64.0"),
        ("pyunpack", "pyunpack"),
        ("python-dateutil", "dateutil"),
        ("chardet", "chardet", "5.1.0"),
        ("Pillow", "PIL"),
        ("openpyxl", "openpyxl", "3.1.0"),
        ("joblib", "joblib", "1.2.0"),
        ("wandb", "wandb", "0.15.0"),
        ("tensorboard", "tensorboard", "2.12.0")
    ]
    
    all_ok = True
    
    for package in utility_packages:
        if len(package) == 3:
            name, import_name, min_version = package
            if not check_package(name, import_name, min_version):
                all_ok = False
        else:
            name, import_name = package
            if not check_package(name, import_name):
                all_ok = False
    
    if not check_patool_special():
        all_ok = False
    
    return all_ok

def test_basic_functionality():
    print("\nTesting Basic Functionality...")
    
    tests_passed = 0
    total_tests = 0
    
    # test Pytorch functions
    try:
        import torch
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("PyTorch tensor operations: Working")
        tests_passed += 1
    except Exception as e:
        print(f"PyTorch tensor operations: Failed - {e}")
    total_tests += 1
    
    # test PyG functions
    try:
        import torch_geometric
        from torch_geometric.data import Data
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        print("PyG graph creation: Working")
        tests_passed += 1
    except Exception as e:
        print(f"PyG graph creation: Failed - {e}")
    total_tests += 1
    
    # test DGL functions
    try:
        import dgl
        import networkx as nx
        g_nx = nx.petersen_graph()
        g_dgl = dgl.from_networkx(g_nx)
        print("DGL graph creation: Working")
        tests_passed += 1
    except Exception as e:
        print(f"DGL graph creation: Failed - {e}")
    total_tests += 1
    
    # test NLP functions
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode("Hello world")
        print("Sentence Transformers: Working")
        tests_passed += 1
    except Exception as e:
        print(f"Sentence Transformers: Failed - {e}")
    total_tests += 1
    
    # test data processing functions
    try:
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = df.sum()
        print("Pandas operations: Working")
        tests_passed += 1
    except Exception as e:
        print(f"Pandas operations: Failed - {e}")
    total_tests += 1
    
    print(f"Basic functionality tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def check_data_directories():
    print("\nChecking Directory Structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"Directory: {dir_path}")
        else:
            print(f"Directory: {dir_path} - Missing")
            all_exist = False
    
    return all_exist

def main():
    print("Starting Comprehensive Environment Verification...")
    print("=" * 60)
    
    # Python version
    print(f"Python Version: {sys.version}")
    print("=" * 60)
    
    # check all packages
    all_checks_passed = True
    
    print("\nChecking Core Packages...")
    all_checks_passed &= check_cuda()
    
    print("\nChecking PyTorch Geometric Packages...")
    all_checks_passed &= check_pyg()
    
    print("\nChecking Data Science Packages...")
    all_checks_passed &= check_data_science_packages()
    
    print("\nChecking NLP Packages...")
    all_checks_passed &= check_nlp_packages()
    
    print("\nChecking Graph Packages...")
    all_checks_passed &= check_graph_packages()
    
    print("\nChecking Utility Packages...")
    all_checks_passed &= check_utility_packages()
    
    # test functions
    all_checks_passed &= test_basic_functionality()
    
    # check directory
    all_checks_passed &= check_data_directories()
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("ALL CHECKS PASSED! Your environment is ready for the project.")
        print("\nNext steps:")
        print("1. Run: python download_data.py")
        print("2. Run: python scripts/data_collection.py") 
        print("3. Run: python scripts/neo4j_import.py")
        print("4. Run: python scripts/han_model.py")
    else:
        print("SOME CHECKS FAILED! Please review the errors above.")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    exit(main())