import requests
import json
from tqdm import tqdm
import os
import zipfile
import gzip
import shutil

class AminerDataDownloader:
    def __init__(self):
        self.raw_dir = "data/raw"
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def download_file(self, url, filename):
        """下载文件并显示进度条"""
        print(f"📥 Downloading from: {url}")
        print(f"📁 Saving to: {filename}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file, tqdm(
                desc=os.path.basename(filename),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=8192):
                    size = file.write(data)
                    bar.update(size)
            
            file_size = os.path.getsize(filename)
            print(f"✅ Downloaded: {file_size:,} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def extract_zip(self, zip_path):
        """解压ZIP文件到raw目录"""
        print(f"📦 Extracting {zip_path}...")
        try:
            if zipfile.is_zipfile(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
                print(f"✅ ZIP extraction successful")
                
                # 解压成功后删除ZIP文件
                try:
                    os.remove(zip_path)
                    print(f"🗑️  Deleted ZIP file: {os.path.basename(zip_path)}")
                except Exception as e:
                    print(f"⚠️  Could not delete ZIP file: {e}")
                
                return True
            elif zip_path.endswith('.gz'):
                return self.extract_gzip(zip_path)
            else:
                print(f"ℹ️  Not a compressed file, skipping extraction")
                return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def extract_gzip(self, gz_path):
        """解压GZIP文件"""
        print(f"📦 Extracting gzip file: {gz_path}")
        try:
            output_path = gz_path.replace('.gz', '')
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✅ Gzip extraction successful: {output_path}")
            
            # 解压成功后删除GZIP文件
            try:
                os.remove(gz_path)
                print(f"🗑️  Deleted GZIP file: {os.path.basename(gz_path)}")
            except Exception as e:
                print(f"⚠️  Could not delete GZIP file: {e}")
            
            return True
        except Exception as e:
            print(f"❌ Gzip extraction failed: {e}")
            return False
    
    def check_existing_data(self):
        """检查是否已有数据文件"""
        if not os.path.exists(self.raw_dir):
            return []
        
        data_files = []
        for file in os.listdir(self.raw_dir):
            if file.endswith(('.json', '.txt', '.gz', '.zip')):
                data_files.append(file)
        return data_files
    
    def create_sample_data(self):
        """创建示例数据文件用于测试"""
        print("\n📝 Creating sample citation data for testing...\n")
        
        sample_data = []
        venues = ["ICML", "NeurIPS", "ICLR", "KDD", "WWW", "AAAI", "IJCAI", "ACL", "EMNLP", "CVPR"]
        topics = [
            "Graph Neural Networks", "Deep Learning", "Natural Language Processing",
            "Computer Vision", "Reinforcement Learning", "Knowledge Graphs",
            "Machine Learning", "Data Mining", "Information Retrieval", "AI Ethics"
        ]
        
        for i in range(200):
            num_authors = 2 + (i % 4)
            num_refs = min(i, 10)
            
            paper = {
                "id": f"paper_{i:04d}",
                "title": f"{topics[i % len(topics)]}: A Study on {topics[(i+3) % len(topics)]}",
                "authors": [
                    {
                        "name": f"Author_{i}_{j}",
                        "id": f"author_{(i*10+j):04d}",
                        "org": f"University {(i+j) % 20}"
                    }
                    for j in range(num_authors)
                ],
                "year": 2015 + (i % 10),
                "venue": venues[i % len(venues)],
                "n_citation": max(0, 50 - i // 4),
                "references": [f"paper_{j:04d}" for j in range(max(0, i-num_refs), i)],
                "abstract": f"This paper presents a comprehensive study on {topics[i % len(topics)].lower()}. "
                           f"We propose novel methods and evaluate them on standard benchmarks. "
                           f"Our approach achieves state-of-the-art results.",
                "keywords": [topics[i % len(topics)], topics[(i+1) % len(topics)], topics[(i+2) % len(topics)]]
            }
            sample_data.append(paper)
        
        sample_file = os.path.join(self.raw_dir, "sample_citation_data.json")
        try:
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(sample_file)
            print(f"✅ Sample data created successfully!")
            print(f"   📄 File: {sample_file}")
            print(f"   📊 Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
            print(f"   📚 Papers: {len(sample_data)}")
            print(f"   👥 Authors: ~{len(sample_data) * 3}")
            print(f"   🔗 Citations: {sum(len(p['references']) for p in sample_data)}")
            return True
        except Exception as e:
            print(f"❌ Failed to create sample data: {e}")
            return False
    
    def download_aminer_dataset(self):
        """下载Aminer数据集"""
        print("\n🎯 Starting Aminer dataset download...")
        
        # 检查现有数据
        existing_files = self.check_existing_data()
        if existing_files:
            print(f"\n✅ Found existing files in {self.raw_dir}:")
            for f in existing_files:
                file_path = os.path.join(self.raw_dir, f)
                file_size = os.path.getsize(file_path)
                print(f"   📄 {f} ({file_size:,} bytes)")
            
            use_existing = input("\n❓ Use existing data? (y/n): ").strip().lower()
            if use_existing == 'y':
                return True
        
        # 尝试下载
        print("\n⚠️  Note: Aminer URLs frequently change.")
        print("📖 Manual download guide:")
        print("   1. Visit: https://www.aminer.cn/data")
        print("   2. Download DBLP or AMiner citation dataset")
        print("   3. Place files in data/raw/\n")
        
        download_urls = [
            ("https://static.aminer.cn/lab-datasets/citation/citation-network1.zip", 
             "citation-network1.zip"),
        ]
        
        for url, filename in download_urls:
            file_path = os.path.join(self.raw_dir, filename)
            
            print(f"📌 Trying to download {filename}...")
            if self.download_file(url, file_path):
                if filename.endswith(('.zip', '.gz')):
                    self.extract_zip(file_path)
                return True
        
        # 询问是否创建测试数据
        print("\n" + "=" * 70)
        print("🔧 Automatic download failed. Create sample test data?")
        print("=" * 70)
        print("This creates a synthetic dataset (200 papers) for testing.")
        
        create_sample = input("\n❓ Create sample data? (y/n): ").strip().lower()
        if create_sample == 'y':
            return self.create_sample_data()
        
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("📥 Aminer Dataset Downloader")
    print("=" * 70)
    
    downloader = AminerDataDownloader()
    success = downloader.download_aminer_dataset()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Download completed successfully!")
        print(f"📁 Data saved to: {downloader.raw_dir}/")
        
        # 显示下载的文件
        files = downloader.check_existing_data()
        if files:
            print(f"\n📋 Files in {downloader.raw_dir}:")
            total_size = 0
            for f in files:
                file_path = os.path.join(downloader.raw_dir, f)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                print(f"   📄 {f}: {file_size:,} bytes")
            print(f"\n📊 Total: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        
    else:
        print("❌ Download failed!")
        print("\n💡 Options:")
        print("   1. Manually download from https://www.aminer.cn/data")
        print("   2. Place data files in data/raw/")
        print("   3. Run this script again and choose to create sample data")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
