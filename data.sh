#!/bin/bash
# Data Download and Processing Script
# 一键完成数据下载和处理

echo "======================================================================"
echo "🚀 Academic Data Download and Processing Pipeline"
echo "======================================================================"

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python first."
    exit 1
fi

# 步骤1: 下载数据
echo ""
echo "📥 Step 1/2: Downloading data..."
echo "----------------------------------------------------------------------"
python download_data.py
DOWNLOAD_STATUS=$?

if [ $DOWNLOAD_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Data download failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# 步骤2: 处理数据
echo ""
echo "🔄 Step 2/2: Processing data..."
echo "----------------------------------------------------------------------"
python process_data.py
PROCESS_STATUS=$?

if [ $PROCESS_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Data processing failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# 完成
echo ""
echo "======================================================================"
echo "✅ Data pipeline completed successfully!"
echo "======================================================================"
echo ""
echo "📁 Data locations:"
echo "   • Raw data: data/raw/"
echo "   • Processed CSV: data/processed/"
echo ""
echo "📋 Next steps:"
echo "   1. Verify CSV files in data/processed/"
echo "   2. Start Neo4j database"
echo "   3. Run: python neo4j_import.py"
echo "   4. Run: python han_model.py"
echo ""
echo "======================================================================"

exit 0
